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
- JSONL traces, one file per episode. Subagent episodes get their own file with parent linkage.

## Install & Run

```bash
cd clawTrace
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python runner.py "Read these three files and summarize each: README.md, implement.md, config.py"
```

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
runner.py
  ├─▶ engine.py      run_episode loop, Episode, SubagentBudget,
  │                    RuntimeContext, ToolExecutor
  ├─▶ subagent.py    spawn_subagent tool (separate file to break
  │                    circular import with engine.run_episode)
  ├─▶ config.py      RunConfig dataclass + DEFAULT_CONFIG dict
  ├─▶ llm.py         OpenAI streaming, run_assistant_turn
  ├─▶ tools.py       ToolDef registry + 6 built-ins
  ├─▶ trace.py       TraceWriter (thread-safe), make_event
  └─▶ vendor/
        providers.py   cheetahclaws' multi-backend streaming,
                         vendored but unused (reserved for vLLM)
```

One step = one assistant turn. One JSONL file per episode. Parallelism comes from a single `ThreadPoolExecutor` inside `ToolExecutor` — not a dedicated subagent pool.

For the full architecture, decisions, and scaling discipline: **[implement.md](implement.md)**.

## Config (18 knobs)

Highlights — see [implement.md §5](implement.md#5-config-knobs-all-18) for the full list.

| Knob | Default | What it changes |
|---|---|---|
| `model` | `"gpt-4o-mini"` | Provider / cost / capability |
| `max_steps` | `8` | Hard loop bound |
| `allow_parallel_tools` | `False` | Permit within-turn tool parallelism |
| `max_parallel_tools` | `4` | Worker cap when parallel is allowed |
| `enable_subagents` | `False` | Filter `spawn_subagent` from LLM-visible tools |
| `subagent_parallelism` | `"serial"` | Subagent write-conflict policy |
| `max_subagent_depth` | `0` | Nesting cap (`0` = unlimited) |
| `runs_per_task` | `1` | Repeated runs per task (planned; not yet wired) |

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

Not built yet (roadmap in [implement.md §11](implement.md#11-whats-next)):

- vLLM backend
- `runs_per_task` loop in runner
- Trace analysis script
- TBT metric
- SWE-bench workload dispenser
- `history_policy` / context compaction
- `subagent_parallelism="worktree"` and `"shared+optimistic"` modes

## What It Differs From

- **`cheetahclaws`**: we borrow the core loop shape and tool registry but skip bridges, permissions, plan mode, checkpointing, UI, MCP, memory, multi-agent manager, compaction. Traces are the goal, not a chat product.
- **Claude Code**: same general mechanism (native tool-calling, per-episode trace file, subagent-as-tool). Different concurrency model (sync + threads vs. async). No permission layer. No streaming UI.

## License

Same as the parent repository.
