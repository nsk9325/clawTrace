# ClawTrace

ClawTrace is a small, trace-first harness for running and profiling agentic LLM workloads.

It is intentionally narrower than `cheetahclaws`.

`cheetahclaws` is an assistant runtime.
`clawTrace` is an execution-and-tracing framework.

The current implementation focuses on:

- a bounded assistant-turn loop
- OpenAI-compatible streamed LLM calls
- native model tool-calling
- JSONL traces as the primary artifact

## Current Architecture

```text
runner.py
  -> engine.py
      -> llm.py
      -> tools.py
      -> trace.py
      -> config.py
```

### `runner.py`

CLI entry point.

Responsibilities:

- accept task input from the command line
- start one episode
- print episode/run ids and trace path

### `engine.py`

Main agentic loop.

Current step definition:

- one `step` = one assistant turn

Current loop shape:

1. initialize message history with one user task
2. call the LLM with history and tool schemas
3. receive one assistant turn:
   - assistant text
   - zero, one, or many tool calls
4. trace the LLM call
5. append the assistant turn to history
6. execute each tool call
7. trace each tool call
8. append tool results to history
9. trace context update and step end
10. stop when there are no more tool calls or `max_steps` is reached

### `llm.py`

OpenAI-compatible streaming layer.

Responsibilities:

- convert neutral history into OpenAI message format
- pass tool schemas to the model
- stream assistant text and tool-call deltas
- reconstruct full tool calls from streamed chunks
- return one completed `AssistantTurn`
- measure client-side timing:
  - `latency_ms`
  - `ttft_ms`
  - `prefill_time_ms`
  - `decode_time_ms`

### `tools.py`

Minimal tool system adapted from `cheetahclaws`.

Current built-in tools:

- `bash`
- `read_file`

Responsibilities:

- expose tool schemas to the LLM
- dispatch tool calls by name
- run the actual tool implementation

### `trace.py`

Trace writer and event builder.

Responsibilities:

- create common event records
- append JSONL events to the trace file

### `config.py`

Minimal runtime configuration.

Current defaults include:

- `max_steps`
- `backend`
- `model`
- `temperature`
- `max_tokens`
- `tool_timeout_s`
- `output_dir`

Current direction for configuration:

- keep the config surface small
- prefer behavior-changing experiment knobs over product-style settings
- add new config only when it changes runtime behavior in a measurable way

## Message Model

The engine now uses a neutral message history modeled after `cheetahclaws`:

```python
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
```

This is the state passed into `llm.py` on each assistant turn.

## Trace Model

Each trace row is one JSON object in a `.jsonl` file.

Current event types:

- `step_start`
- `llm_call`
- `tool_call`
- `context_update`
- `step_end`
- `episode_end`

### Current `llm_call` fields

- `backend`
- `model`
- `latency_ms`
- `prompt_tokens`
- `completion_tokens`
- `ttft_ms`
- `prefill_time_ms`
- `decode_time_ms`
- `measurement`
- `assistant_text_preview`
- `tool_call_count`

### Current `tool_call` fields

- `tool_call_id`
- `tool_name`
- `params`
- `latency_ms`
- `exit_status`
- `result_preview`

## How It Differs From `cheetahclaws`

ClawTrace intentionally resembles the core loop shape of `cheetahclaws`, but not the whole product/runtime.

Borrowed ideas:

- neutral message history
- assistant-turn model
- streamed OpenAI-compatible call path
- streamed tool-call reconstruction
- small tool registry pattern

Intentionally not carried over:

- bridges and chat integrations
- permission UX
- plan mode
- runtime globals
- context compaction
- checkpointing
- subagent framework
- rich user-facing streaming UI

So the rule is:

- `cheetahclaws`: runtime-first
- `clawTrace`: trace-first

## What We Discarded

The project started with a narrower prototype that asked the LLM for one JSON action like:

```json
{"tool_name": "...", "params": {...}}
```

That path has been discarded for the actual runtime loop.

The current loop now uses native model tool-calling instead.

We also removed the mock runtime path.
Tests now stay offline by:

- testing LLM helper functions directly
- using fake assistant turns in engine tests

## Current Status

Working now:

- OpenAI-compatible streamed assistant turns
- native tool-calling reconstruction
- bounded multi-step loop
- per-event JSONL traces
- `bash` and `read_file` tools
- direct-run local tests

Not built yet:

- vLLM backend
- repeated-run experiment runner
- offline analysis script
- real context policies
- parallel tool execution
- subagents

## Configuration Direction

The most important future configuration knobs are expected to be:

- `model`
- `backend`
- `max_steps`
- `temperature`
- `max_tokens`
- `tool_timeout_s`

Likely next experiment knobs:

- `enable_subagents`
- `max_subagents`
- `subagent_strategy`
- `allow_parallel_tools`
- `history_policy`
- `runs_per_task`

These are considered good knobs because they change agent behavior in ways that are worth tracing and comparing.

We do not want a large product-style config surface.

The rule is:

- add config when it changes execution strategy, cost, latency, or trace shape
- avoid config that only exists for convenience unless it is truly needed

### Suggested Future Config Shape

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

## Profiling Direction

The current tracing already records:

- LLM latency
- prompt/completion tokens
- TTFT
- prefill approximation
- decode time
- per-tool latency

The next profiling targets should be:

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

Very useful derived metrics later:

- LLM time fraction vs tool time fraction
- context growth vs latency
- steps to completion
- model choice vs latency/cost
- subagent enabled vs disabled overhead
- parallel vs serial execution overhead

### TBT

`TBT` is still not implemented.

Planned meaning:

- average time between streamed token or chunk arrivals after the first token

This will require recording per-chunk arrival timestamps during streaming.

## Workload Direction

The intended real workload source is `SWE-bench`.

ClawTrace should not treat SWE-bench instances as raw prompts passed directly into the engine.
Instead, the plan is to introduce a configurable workload dispenser layer between the dataset and the runtime.

Conceptually:

```text
SWE-bench instance
  -> workload dispenser
      -> normalized task input / initial messages
          -> runner / engine
```

Why this matters:

- different workload shapes should be measurable
- prompt size and context composition should be controlled intentionally
- local backends such as `vLLM` may need deliberately heavier inputs to stress GPU behavior
- the same underlying task may need multiple renderings for comparison experiments

The workload dispenser should eventually control things like:

- how much task context is included
- whether supporting context is included
- how large the rendered input becomes
- how aggressively the workload stresses the model/backend

The point is not only task delivery, but workload shaping for profiling.

This is important for experiments involving:

- input token differences
- prompt complexity differences
- model comparisons
- local inference pressure such as GPU saturation under `vLLM`

## Running

From inside `clawTrace`:

```bash
source .venv/bin/activate
export OPENAI_API_KEY=your_key_here
python runner.py "Read the README file and summarize the project"
```

After the run:

```bash
ls -lt traces
cat traces/<episode_file>.jsonl
```

## Tests

Current direct-run tests:

```bash
python tests/test.py
python tests/test_config.py
python tests/test_tools.py
python tests/test_llm.py
python tests/test_engine.py
```

## Where We Are Heading

Near-term priorities:

1. validate the live OpenAI tool-calling path carefully
2. add vLLM using the same assistant-turn and trace model
3. add an analysis script for trace summaries
4. add experiment-facing config knobs such as model selection and subagent toggles
5. design the workload dispenser for SWE-bench-driven experiments
6. decide how far to simplify the runtime for a strict experimental setup

The main design constraint remains:

keep the loop explicit, bounded, and easy to trace.
