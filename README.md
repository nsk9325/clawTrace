# ClawTrace

clawTrace is end-to-end agentic workload tracer including
- **A SWE-bench workload dispenser**
- **A highly-configurable agentic wrapper**, leaving a JSONL trace file
- **A Trace analyzer**, creating the main artifacts

Given a workload manually or by the SWE-bench workload dispenser, it is handled by the agentic loop, then leaves a JSONL trace file with every LLM call, tool call, and subagent spawn timestamped and attributed. It can be analyzed manually or summarized by the analyzer.

![Architecture](/img/Architecture.png)

## Install & Run

### Install & set the API key

```bash
cd clawTrace
source .venv/bin/activate
pip install -r requirements.txt

# OpenAI backend (default):
export OPENAI_API_KEY=sk-...

# Or any OpenAI-compatible backend (vLLM, LiteLLM proxy, etc.):
export CUSTOM_API_KEY=<key-or-EMPTY>
```

### Run SWE-bench instances (dispense → profile → analyze → visualize)

```bash
# After downloading swebench_lite.jsonl and pre-cloning a repo to
# ./swebench-repos/<instance_id>/ at the right base_commit:
./run_swebench.sh --instance-id astropy__astropy-12907 --config '{"max_steps":20}'
```

### Run manually
```bash
python runner.py "Read these three files and summarize each: README.md, implement.md, config.py"
```

### Run against vLLM (or any OpenAI-compatible endpoint)

clawTrace's `custom` backend talks to any OpenAI-compatible HTTP endpoint.
vLLM is the primary target.

**1. Launch vLLM** (separate terminal / pod / box). Wait for `Application
startup complete`:

```bash
vllm serve <hf-id> \
  --tool-call-parser <name> \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  --port 8000
```

Three flags are load-bearing:
- `--tool-call-parser` — vLLM's server-side adapter from the model's native
  tool-call syntax to OpenAI shape. Must match the model family:

  | Model family | `--tool-call-parser` |
  |---|---|
  | Llama-3.1 / 3.2 / 3.3 | `llama3_json` |
  | Qwen2.5 (Instruct / Coder) | `hermes` |
  | Hermes-3 | `hermes` |
  | Mistral-Nemo / Mistral-Small | `mistral` |
  | DeepSeek-V2.5 / V3 | `deepseek_v3` |

- `--enable-auto-tool-choice` — required (clawTrace hardcodes `tool_choice="auto"`).
- `--enable-prefix-caching` — required for `cached_tokens` to populate.

**2. Set the API key env var** — vLLM ignores the *value* but the OpenAI SDK
requires *some* non-empty string. If vLLM was launched with `--api-key X`,
match it; otherwise any string works:

```bash
export CUSTOM_API_KEY=EMPTY
```

**3. Run clawTrace pointing at the vLLM URL:**

```bash
python runner.py "<task>" --config '{
  "backend": "custom",
  "model": "<exact-hf-id>",
  "custom_base_url": "http://<host>:8000/v1",
  "max_steps": 5
}'
```

`backend="custom"` is **required** when the model id contains a slash.
clawTrace doesn't parse slashes for routing — `Qwen/Qwen2.5-7B-Instruct`
is a literal HF model id, not a `provider/model` shortcut. Without
`backend` set, you'd hit `Backend 'Qwen' is not implemented yet`.

**RunPod gotchas** (encountered in our smoke tests):

- Use the **HTTP Service Ports** slot for port 8000, not "TCP Ports". The
  resulting proxy URL is `https://<POD_ID>-8000.proxy.runpod.net` — append
  `/v1` for `custom_base_url`.
- The pre-built RunPod vLLM template often launches with `--api-key sk-...`.
  Either pass that exact key via `custom_api_key` config / `CUSTOM_API_KEY`
  env, or edit the pod's "Container Start Command" to drop `--api-key`.
- Default `--max-model-len` in templates is small (commonly 8K). SWE-bench
  runs accumulate tool results fast and can blow past it. Bump in the start
  command (Qwen2.5-7B supports 32K natively) or cap `max_steps` low.
- `pkill -f "vllm serve"` will kill the container's PID 1 if vLLM is
  launched via docker-init; the container restarts. To stop just the
  Python worker, use `pkill -f "python.*vllm"` instead.

### Pass configs at runtime

Both `runner.py` and `profile_swebench.py` (and `run_swebench.sh` by
forwarding) accept `--config '<json>'`. The JSON dict is merged on top of
`RunConfig` defaults at `run_episode` entry:

```bash
python runner.py "task" --config '{"max_steps": 10, "model": "gpt-4o"}'
./run_swebench.sh --instance-id <id> --config '{"backend":"custom","model":"...","custom_base_url":"..."}'
```

See [Main Configurations](#main-configurations) for the full knob list.

### Analyze, visualize, or token-trace a run

`run_swebench.sh` runs all three on each trace it produces. To run them
manually on any trace:

**Analyzer** — per-episode metrics, full cfg + workload dump, recursive
subagent tree. Auto-saves `<stem>.analysis.txt` next to the trace:
```bash
python analyzer.py traces/episode_<root_id>/episode_<root_id>.jsonl
```

**Visualizer** — server/client Gantt chart with LLM activity (TTFT prefill
+ decode) on a server lane and tool execution on a client lane. Subagents
appear as a hatched span on the parent's client lane with their own
lanes nested below. Emits `<stem>.gantt.png` + `<stem>.gantt.json`
(machine-readable timeline for re-rendering):
```bash
python visualizer.py traces/episode_<root_id>/episode_<root_id>.jsonl
# Optional: --detail-tools all|none|<comma-list>   (default: bash)
```

**Token trace** — flattened per-LLM-call JSONL with `t_ms`, `ttft_ms`,
`duration_ms`, `in_tokens`, `out_tokens`, `cached_tokens`. Useful for
inspecting prefix-cache warm-up across an episode (step 0 cold, step 1+
warm). Emits `<stem>.tokens.jsonl`:
```bash
python token_trace_gen.py traces/episode_<root_id>/episode_<root_id>.jsonl
```

## LLM Backend

- Most of the LLM backend is lifted from cheetahclaws.
- Supports the OpenAI API (`backend="openai"`) and any OpenAI-compatible
  HTTP endpoint via `backend="custom"` — vLLM, LiteLLM proxy, llama.cpp
  server, etc. Same OpenAI-format wire shape on both paths; vLLM normalizes
  per-model tool-call syntax server-side via `--tool-call-parser`.
- vLLM-native richer profiling (per-chunk TBT, Prometheus `/metrics`
  snapshots at episode boundaries, real prefill split) is not yet wired —
  see [Implementation Status](#implementation-status).

## TMIs

- Per-step, per-tool timing: `latency_ms`, `ttft_ms`, `prefill_time_ms`, `decode_time_ms`, `started_at_ms` / `ended_at_ms` on every tool call.
- **Optional parallel tool execution** — gated by a safety rule: enabled only when every tool in the batch is `concurrent_safe=True`. 
- **Optional subagent spawning** — recursive, with four budget caps (total / per-parent / depth / concurrent) and a `subagent_parallelism` mode knob for write-conflict policy.
- 6 built-in tools: `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`.
- **Configurable system prompts** — registry-based (`"agent"` / `"minimal"`) or raw override, with config-aware sections that toggle on `allow_parallel_tools` / `enable_subagents`. Subagents always get the minimal prompt.
- JSONL traces, one file per episode. Subagent episodes get their own file with parent linkage.
- **Self-describing traces** — every trace's first event is `episode_start` with task input, model, system prompt name, full config snapshot, and (when driven by the dispenser) workload metadata.
- **SWE-bench dispenser** — drive the agent against real bug-fix instances, capture trace + diff + predictions file per instance.

## Main Configurations

| Knob | Default | What it changes |
|---|---|---|
| `backend` | `"openai"` | Provider routing: `"openai"` or `"custom"` (any OpenAI-compatible endpoint) |
| `model` | `"gpt-4o-mini"` | Model id sent to the API verbatim |
| `max_steps` | `20` | Hard loop bound |
| `system_prompt` | `"agent"` | Registry preset (`"agent"`, `"minimal"`) or raw override |
| `append_system_prompt` | `""` | Free-form text appended after the resolved prompt |
| `allow_parallel_tools` | `False` | Permit within-turn tool parallelism |
| `max_parallel_tools` | `4` | Worker cap when parallel is allowed |
| `enable_subagents` | `True` | Filter `spawn_subagent` from LLM-visible tools |
| `subagent_parallelism` | `"serial"` | Subagent write-conflict policy |
| `max_subagent_depth` | `1` | Nesting cap (`0` = unlimited) |
| `max_subagents_total` | `2` | Lifetime cap across the whole root tree (`0` = unlimited) |
| `runs_per_task` | `1` | Repeated runs per task; runner loops serially for variance measurement |
| `openai_base_url` | `""` | Override OpenAI base URL (Azure, proxies, etc.) |
| `custom_base_url` | `""` | Endpoint URL when `backend="custom"` (e.g. `http://host:8000/v1`) |
| `custom_api_key` | `""` | API key for the custom endpoint; falls back to `CUSTOM_API_KEY` env var |
| `vllm_metrics_url` | `""` | Reserved for future Phase 2 Prometheus snapshots; unread today |

## Implementation Status

What Currently Works

- Parallel tool execution (gated by all-concurrent-safe)
- Subagents with 4-cap budget + `serial` / `shared` parallelism modes
- Per-root-episode trace dir layout: parent + subagent traces + dispenser/analyzer/visualizer artifacts as siblings in `traces/episode_<root_id>/`
- Configurable system prompts with environment / git substitutions and config-aware sections
- SWE-bench workload dispenser + profile CLI; emits trace + diff + predictions file per instance
- Trace analyzer (`analyzer.py`): per-episode metrics, recursive subagent tree, full cfg + workload dump, auto-saves `.analysis.txt`
- Trace visualizer (`visualizer.py`): server/client Gantt PNG + JSON sidecar, follows subagents into nested lanes
- Token trace generator (`token_trace_gen.py`): flattened per-LLM-call JSONL with token volumes + cache hits
- End-to-end shell wrapper (`run_swebench.sh`): dispense → profile → analyze → visualize → token-trace each trace
- Single-source-of-truth config: `RunConfig` dataclass; CLI `--config '<json>'` override on every entry point
- vLLM / OpenAI-compatible backend (`backend="custom"`): typed config fields + CLI flag; clawTrace stays OpenAI-format on the wire

Not built yet

- vLLM-native metrics (Phase 2): per-chunk TBT, Prometheus `/metrics` snapshots at episode boundaries, real prefill/decode split
- `history_policy` / context compaction (real cost: 169K input tokens observed in one 20-step SWE-bench run)
- Per-instance Python env setup for SWE-bench (agents currently can't run target-repo tests, so they edit without feedback)
- `subagent_parallelism="worktree"` and `"shared+optimistic"` modes
- `--eval` flag for `profile_swebench.py` (predictions are emitted; harness call + Docker is the missing part)

## What It Differs From

- **`cheetahclaws`**: we borrow the core loop shape and tool registry but skip bridges, permissions, plan mode, checkpointing, UI, MCP, memory, multi-agent manager, compaction. Traces are the goal, not a chat product.
- **Claude Code**: same general mechanism (native tool-calling, per-episode trace file, subagent-as-tool). Different concurrency model (sync + threads vs. async). No permission layer. No streaming UI.
