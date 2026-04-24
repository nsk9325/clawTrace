# ClawTrace Implementation Notes

This file is the working handoff note for future sessions.

It should answer:

- what ClawTrace is
- what has already been implemented
- what design decisions were made
- what was intentionally discarded
- what the next likely steps are

## Project Intent

ClawTrace is a small framework for profiling agentic workloads.

The main artifact is the trace, not the chat experience.

Design goals:

- bounded execution
- explicit step boundaries
- observable LLM and tool activity
- low runtime complexity

Non-goals for now:

- bridges and chat UI
- permission UX
- plan mode
- subagent orchestration
- heavy memory systems

## Current Architecture

```text
runner.py
  -> engine.py
      -> llm.py
      -> tools.py
      -> trace.py
      -> config.py
```

### `engine.py`

This is the main agentic loop.

Current meaning of a step:

- one `step` = one assistant turn

Current loop behavior:

1. start history with one user task
2. call `llm.run_assistant_turn(...)`
3. trace the `llm_call`
4. append assistant turn to history
5. execute all returned tool calls
6. trace each `tool_call`
7. append tool results to history
8. trace `context_update`
9. trace `step_end`
10. stop when there are no more tool calls or `max_steps` is hit

This is now intentionally closer to `cheetahclaws` than the earlier single-action prototype.

### `llm.py`

Current role:

- OpenAI-compatible streamed assistant-turn execution

Important implementation details:

- uses neutral history -> OpenAI message conversion
- sends tool schemas with `tool_choice="auto"`
- reconstructs tool calls from streamed `delta.tool_calls`
- preserves streamed `finish_reason` when the backend provides it for trace analysis
- returns one `AssistantTurn` dataclass
- records client-side timing:
  - `latency_ms`
  - `ttft_ms`
  - `prefill_time_ms`
  - `decode_time_ms`

### `tools.py`

Current role:

- minimal tool registry + dispatch
- source of LLM-visible tool schemas

Current tools:

- `bash`
- `read_file`

Adapted from `cheetahclaws`, but heavily reduced.

### `trace.py`

Current role:

- JSONL event writer
- common event builder

### `config.py`

Current role:

- small defaults
- JSON load/save helpers

Current design direction:

- keep config small
- prioritize experiment knobs over convenience knobs
- only add config when it changes runtime behavior in a measurable way

### `runner.py`

Current role:

- simplest CLI entry point for one episode

## Message Model

Current neutral history format:

```python
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
```

This was adopted to match the loop shape of `cheetahclaws` without copying its full runtime.

## Trace Model

Current event types:

- `step_start`
- `llm_call`
- `tool_call`
- `context_update`
- `step_end`
- `episode_end`

Current trace file policy:

- one `.jsonl` file per episode

Current `llm_call` fields include:

- `backend`
- `model`
- `latency_ms`
- `prompt_tokens`
- `completion_tokens`
- `ttft_ms`
- `prefill_time_ms`
- `decode_time_ms`
- `measurement`
- `finish_reason`
- `assistant_text_preview`
- `tool_call_count`

Current `tool_call` fields include:

- `tool_call_id`
- `tool_name`
- `params`
- `latency_ms`
- `exit_status`
- `result_preview`

## What Changed During Implementation

### Earlier Prototype

Earlier, the runtime used a much narrower contract:

- LLM returns one JSON action
- engine parses that action
- one tool runs
- stop

That path is no longer the real runtime model.

### Current Runtime Model

Now the runtime is:

- assistant turn
- native tool calls from the model
- append tool results to history
- continue until no tool calls remain

This is the correct direction for profiling real agentic behavior.

## What We Borrowed From `cheetahclaws`

Borrowed directly or closely:

- neutral message-history idea
- assistant-turn abstraction
- streamed OpenAI-compatible call pattern
- streamed tool-call reconstruction
- tool registry shape

Borrowed but simplified:

- provider structure
- tool schema exposure
- engine loop shape

Intentionally not borrowed:

- permissions
- bridges
- plan mode
- compaction
- checkpointing
- subagent manager
- runtime globals

## What We Discarded

Discarded decisions:

- mock runtime path
- single JSON-action runtime model
- treating a step as a single parsed action

Retained only for tests:

- offline verification using fake assistant turns

## Testing Strategy

Because the real runtime now depends on live OpenAI tool-calling, local tests avoid live API calls.

Current strategy:

- `tests/test_llm.py`
  - test helper functions only
- `tests/test_engine.py`
  - inject fake assistant turns
  - verify multi-step trace behavior offline
- `tests/test_tools.py`
  - verify tool dispatch
- `tests/test_config.py`
  - verify config behavior
- `tests/test.py`
  - verify trace writer basics

## Current Known Reality

What is implemented:

- bounded assistant-turn loop
- native model tool-calling reconstruction
- per-tool tracing
- per-turn LLM timing
- OpenAI-compatible runtime

What is still missing:

- vLLM backend
- real repeated-run experiment driver
- trace analysis script
- TBT calculation
- stronger stop/failure handling
- context policies

## Configuration Direction

Planned high-value config knobs:

- `model`
- `backend`
- `max_steps`
- `temperature`
- `max_tokens`
- `tool_timeout_s`

Likely next behavior knobs:

- `enable_subagents`
- `max_subagents`
- `subagent_strategy`
- `allow_parallel_tools`
- `history_policy`
- `runs_per_task`

Why these are good:

- they change execution strategy
- they change latency/cost behavior
- they change trace structure in analyzable ways

What to avoid:

- large product-style config surfaces
- convenience-only config unless it becomes necessary

Suggested target shape:

```python
DEFAULT_CONFIG = {
    "backend": "openai",
    "model": "gpt-4o-mini",
    "max_steps": 8,
    "temperature": 0.0,
    "max_tokens": 512,
    "tool_timeout_s": 30,
    "enable_subagents": False,
    "max_subagents": 0,
    "allow_parallel_tools": False,
    "history_policy": "full",
    "runs_per_task": 1,
    "output_dir": "traces",
}
```

## Profiling Roadmap

Already captured now:

- `latency_ms`
- `prompt_tokens`
- `completion_tokens`
- `ttft_ms`
- `prefill_time_ms`
- `decode_time_ms`
- per-tool latency
- per-tool status

High-value next metrics:

- `llm_call_count`
- `tool_call_count`
- `tool_call_count_by_name`
- `total_llm_time_ms`
- `total_tool_time_ms`
- `total_wall_time_ms`
- `total_prompt_tokens`
- `total_completion_tokens`
- `history_length_per_step`
- `prompt_tokens_per_step`
- `completion_tokens_per_step`
- `tool_calls_per_step`
- `assistant_text_length_per_step`

High-value derived comparisons later:

- LLM time fraction vs tool time fraction
- context growth vs latency
- steps to completion
- model comparison by latency/cost
- subagents on vs off
- parallel vs serial execution

### TBT

`TBT` is not implemented yet.

Planned interpretation:

- average time between streamed token or chunk arrivals after the first token

To support this, `llm.py` will need to record per-chunk arrival timestamps.

## Workload Direction

The intended real workload source is `SWE-bench`.

Current runtime input is still simple task text, but this is not the intended long-term workload interface.

Planned direction:

- add a configurable workload dispenser between dataset instances and the runtime
- normalize dataset instances into task inputs or initial message histories
- allow multiple workload renderings for the same underlying task

Conceptually:

```text
SWE-bench instance
  -> workload dispenser
      -> normalized task input / initial messages
          -> runner / engine
```

Why this matters for profiling:

- workload differences should be measurable, not accidental
- input size should be controlled intentionally
- context composition should be configurable
- local backends such as `vLLM` may require heavier rendered workloads to apply meaningful pressure to GPU execution

The workload dispenser should eventually become the place where we control:

- how much task context is included
- how the task is rendered into model input
- how aggressively workloads stress the backend
- how workload variants are compared in profiling runs

This is important for studying:

- input-token effects
- workload-shape effects
- model differences under the same task
- local inference pressure under `vLLM` or other local serving setups

## How To Run

From inside `clawTrace`:

```bash
source .venv/bin/activate
export OPENAI_API_KEY=your_key_here
python runner.py "Read the README file and summarize the project"
```

Then inspect traces:

```bash
ls -lt traces
cat traces/<episode_file>.jsonl
```

## Next Likely Steps

1. Validate the live OpenAI tool-calling loop carefully on a few tasks.
2. Add `vLLM` using the same assistant-turn contract.
3. Add an analysis script to summarize episode traces.
4. Add experiment-facing config knobs such as model selection and subagent toggles.
5. Design and implement the SWE-bench workload dispenser.
6. Decide whether `tools.py` should stay registry-based or be simplified to fixed dispatch.
7. Add `TBT` if chunk timing is useful enough.

## Guidance For Future Sessions

If resuming later, assume:

- the real runtime path is the assistant-turn loop, not the old JSON-action path
- `engine.py` is the main execution/control layer
- `llm.py` is the first place to change if model behavior looks wrong
- `trace.py` output is the main truth source for debugging behavior
- keep the system small and explicit rather than feature-rich

If changing architecture, preserve these invariants:

- one step = one assistant turn
- traces remain JSONL and easy to inspect
- tool calls are explicit events
- history format stays neutral and simple
