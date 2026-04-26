# ClawTrace

A configurable wrapper with profiling around agentic LLM workloads. The primary artifact is the **trace** — a JSONL file per episode with every LLM call, tool call, and subagent spawn timestamped and attributed.

Not a chat product. Not a general-purpose agent runtime. A profiling harness.

## Scope

`cheetahclaws` is an assistant runtime. `clawTrace` is an execution-and-tracing framework. We **vendor** `cheetahclaws`' provider code for future vLLM support but reimplement the loop ourselves so the trace shape stays under our control.

## Features

- Bounded assistant-turn loop. `max_steps` is hard.
- Native model tool-calling (OpenAI-compatible streaming today).
- Per-step, per-tool timing: `latency_ms`, `ttft_ms`, `prefill_time_ms`, `decode_time_ms`, `started_at_ms` / `ended_at_ms` on every tool call.
- **Optional parallel tool execution** — gated by a safety rule: enabled only when every tool in the batch is `concurrent_safe=True`. See [implement.md](implement.md#d6-parallel-tool-calls--option-a-vs-option-b-gate).
- **Optional subagent spawning** — recursive, with four budget caps (total / per-parent / depth / concurrent) and a `subagent_parallelism` mode knob for write-conflict policy.
- 6 built-in tools: `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`.
- **Configurable system prompts** — registry-based (`"agent"` / `"minimal"`) or raw override, with config-aware sections that toggle on `allow_parallel_tools` / `enable_subagents`. Subagents always get the minimal prompt.
- JSONL traces, one file per episode. Subagent episodes get their own file with parent linkage.
- **Self-describing traces** — every trace's first event is `episode_start` with task input, model, system prompt name, full config snapshot, and (when driven by the dispenser) workload metadata.
- **SWE-bench dispenser** — drive the agent against real bug-fix instances, capture trace + diff + predictions file per instance.

## Install & Run

```bash
cd clawTrace
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python runner.py "Read these three files and summarize each: README.md, implement.md, config.py"
```

Drive against SWE-bench instances:

```bash
# After downloading swebench_lite.jsonl and pre-cloning a repo to
# ./swebench-repos/<instance_id>/ at the right base_commit:
python profile_swebench.py \
  --instances swebench_lite.jsonl \
  --repos-dir ./swebench-repos \
  --instance-id astropy__astropy-12907 \
  --config '{"max_steps":20}'
```

See [workload.md](workload.md) for the dispenser's design + setup guide.

Inspect the trace:

```bash
ls -lt traces/ | head
python -c "
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(r['type'], r.get('step_id'), r.get('tool_name', r.get('model','')))
" traces/episode_<hex>.jsonl
```

## Architecture

```text
runner.py                  Plain CLI: one task → one episode
profile_swebench.py        SWE-bench CLI: many instances → many episodes
  │
  ├─▶ engine.py            run_episode loop, Episode, SubagentBudget,
  │                          RuntimeContext, ToolExecutor
  ├─▶ subagent.py          spawn_subagent tool (separate file to break
  │                          circular import with engine.run_episode)
  ├─▶ memory.py            Memory class (history list + system prompt)
  ├─▶ prompts.py           build_system_prompt(cfg, is_subagent), preset
  │                          registry, env / git substitutions
  ├─▶ swebench_dispenser.py  Workload, characterize, select, reset_repo,
  │                            capture_diff, write_predictions
  ├─▶ config.py            RunConfig dataclass + DEFAULT_CONFIG dict
  ├─▶ llm.py               OpenAI streaming, run_assistant_turn(memory, ...)
  ├─▶ tools.py             ToolDef registry + 6 built-ins
  ├─▶ trace.py             TraceWriter (thread-safe), make_event
  └─▶ vendor/
        providers.py       cheetahclaws' multi-backend streaming,
                             vendored but unused (reserved for vLLM)
```

One step = one assistant turn. One JSONL file per episode. Parallelism comes from a single `ThreadPoolExecutor` inside `ToolExecutor` — not a dedicated subagent pool.

For the full architecture, decisions, and scaling discipline: **[implement.md](implement.md)**.

## Config (20 knobs)

Highlights — see [implement.md §7](implement.md#7-config-knobs-all-20) for the full list.

| Knob | Default | What it changes |
|---|---|---|
| `model` | `"gpt-4o-mini"` | Provider / cost / capability |
| `max_steps` | `20` | Hard loop bound |
| `system_prompt` | `"agent"` | Registry preset (`"agent"`, `"minimal"`) or raw override |
| `append_system_prompt` | `""` | Free-form text appended after the resolved prompt |
| `allow_parallel_tools` | `False` | Permit within-turn tool parallelism |
| `max_parallel_tools` | `4` | Worker cap when parallel is allowed |
| `enable_subagents` | `False` | Filter `spawn_subagent` from LLM-visible tools |
| `subagent_parallelism` | `"serial"` | Subagent write-conflict policy |
| `max_subagent_depth` | `0` | Nesting cap (`0` = unlimited) |
| `runs_per_task` | `1` | Repeated runs per task; runner loops serially for variance measurement |

Rule: a new knob earns its place when it changes execution strategy, latency, cost, or trace shape. Convenience-only knobs are rejected.

## Tests

```bash
python -m pytest tests/
```

Live runs are tested manually with `runner.py`; offline tests use fake assistant turns.

## Status

Working:

- OpenAI-compatible streamed assistant turns with finish_reason preserved
- Native tool-calling reconstruction from streamed deltas
- Bounded multi-step loop
- Parallel tool execution (gated by all-concurrent-safe)
- Subagents with 4-cap budget + `serial` / `shared` parallelism modes
- Per-episode JSONL traces with parent linkage on every event
- Self-describing traces via `episode_start` (task, model, prompt name, cfg snapshot, workload metadata)
- Configurable system prompts with environment / git substitutions and config-aware sections
- Repeated-run driver (`runs_per_task`) with per-run and aggregate wall-time reporting
- SWE-bench workload dispenser + profile CLI; emits trace + diff + predictions file per instance

Not built yet (roadmap in [implement.md §15](implement.md#15-whats-next)):

- Trace analysis script (now the gating piece for actually using dispenser traces)
- vLLM backend
- TBT metric
- `history_policy` / context compaction
- `subagent_parallelism="worktree"` and `"shared+optimistic"` modes
- `--eval` flag for `profile_swebench.py` (predictions are emitted; harness call + Docker is the missing part)

## What It Differs From

- **`cheetahclaws`**: we borrow the core loop shape and tool registry but skip bridges, permissions, plan mode, checkpointing, UI, MCP, memory, multi-agent manager, compaction. Traces are the goal, not a chat product.
- **Claude Code**: same general mechanism (native tool-calling, per-episode trace file, subagent-as-tool). Different concurrency model (sync + threads vs. async). No permission layer. No streaming UI.

## License

Same as the parent repository.
