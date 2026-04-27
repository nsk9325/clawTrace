# ClawTrace

clawTrace is end-to-end agentic workload tracer including
- **A SWE-bench workload dispenser**
- **A highly-configurable agentic wrapper**, leaving a JSONL trace file
- **A Trace analyzer**, creating the main artifacts

Given a workload manually or by the SWE-bench workload dispenser, it is handled by the agentic loop, then leaves a JSONL trace file with every LLM call, tool call, and subagent spawn timestamped and attributed. It can be analyzed manually or summarized by the analyzer.

![Architecture](/img/Architecture.png)

## Install & Run

### Install & Export Token (OPENAI)
```bash
cd clawTrace
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

### Run SWE-bench instances (dispense → profile → analyze)

```bash
# After downloading swebench_lite.jsonl and pre-cloning a repo to
# ./swebench-repos/<instance_id>/ at the right base_commit:
./run_swebench.sh --instance-id astropy__astropy-12907 --config '{"max_steps":20}'
```

### Run manually
```bash
python runner.py "Read these three files and summarize each: README.md, implement.md, config.py"
```

### Run Analyzer manually

```bash
python analyzer.py traces/episode_<root_id>/episode_<root_id>.jsonl
# Prints summary + workload + config + subagent tree, and saves the
# same to traces/episode_<root_id>/episode_<root_id>.analysis.txt
```

## LLM Backend

- Most of the LLM backend is lifted from cheetahclaws.
- currently supports OPENAI API
- In future, will support local vLLM serving with richer profiling

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
| `model` | `"gpt-4o-mini"` | Provider / cost / capability |
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

## Implementation Status

What Currently Works

- Parallel tool execution (gated by all-concurrent-safe)
- Subagents with 4-cap budget + `serial` / `shared` parallelism modes
- Per-root-episode trace dir layout: parent + subagent traces + dispenser/analyzer artifacts as siblings in `traces/episode_<root_id>/
- Configurable system prompts with environment / git substitutions and config-aware sections
- SWE-bench workload dispenser + profile CLI; emits trace + diff + predictions file per instance
- Trace analyzer (`analyzer.py`): per-episode metrics, recursive subagent tree, full cfg + workload dump, auto-saves `.analysis.txt`
- End-to-end shell wrapper (`run_swebench.sh`): dispense → profile → analyze each trace
- Single-source-of-truth config: `RunConfig` dataclass

Not built yet

- vLLM backend
- `history_policy` / context compaction (real cost: 169K input tokens observed in one 20-step SWE-bench run)
- Per-instance Python env setup for SWE-bench (agents currently can't run target-repo tests, so they edit without feedback)
- TBT metric (maybe needs a vLLM backend)
- `subagent_parallelism="worktree"` and `"shared+optimistic"` modes
- `--eval` flag for `profile_swebench.py` (predictions are emitted; harness call + Docker is the missing part)

## What It Differs From

- **`cheetahclaws`**: we borrow the core loop shape and tool registry but skip bridges, permissions, plan mode, checkpointing, UI, MCP, memory, multi-agent manager, compaction. Traces are the goal, not a chat product.
- **Claude Code**: same general mechanism (native tool-calling, per-episode trace file, subagent-as-tool). Different concurrency model (sync + threads vs. async). No permission layer. No streaming UI.
