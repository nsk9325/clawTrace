# ClawTrace Implementation Notes

Working handoff doc. Answers:

- what ClawTrace is
- what's built, where each piece came from
- the decisions we weighed and how we picked
- architectural invariants
- the concurrency model's non-obvious corners
- failure modes to debug against
- what to be careful about
- how to scale without drowning in `if cfg.get(...)` branches
- environment / compatibility notes
- what's next

**Section map:**

1. Project Intent
2. Current Architecture — module-by-module
3. Code Provenance — cheetahclaws vs. ours vs. modified
4. Decisions We Weighed (D1–D12)
5. The Mental Model — step loop, parallelism, trace invariants
6. Concurrency Model Details — thread topology, lock scope, ordering, traps
7. Config Knobs (all 20)
8. Trace Model — event types & field groupings
9. Failure Modes & Potential Errors — enumerated
10. What To Be Careful About
11. How To Scale — 6 rules
12. Testing Strategy
13. Environment & Compatibility
14. What We Discarded
15. What's Next
16. How To Run
17. Guidance For Future Sessions

## 1. Project Intent

ClawTrace is a **configurable wrapper with profiling** around agentic LLM workloads.

The primary artifact is the **trace**, not a chat experience.

Design goals:

- bounded execution
- explicit step boundaries
- observable LLM and tool activity
- low runtime complexity
- reusable components from `cheetahclaws` where they already solve the problem

Non-goals:

- chat UI / bridges / voice
- permission UX
- plan mode
- context compaction (deferred as a future knob)
- heavy memory systems
- runtime globals

## 2. Current Architecture

```text
ENTRY POINTS (project root)
  runner.py              Plain CLI: one task_input → one episode
  profile_swebench.py    SWE-bench CLI: many instances → many episodes
  analyzer.py            Per-trace metrics + subagent tree → analysis.txt
  run_swebench.sh        Bash wrapper: dispense → profile → analyze

CORE LOOP
  engine.py              Episode, RuntimeContext, SubagentBudget,
                           ToolExecutor, run_episode
  subagent.py            spawn_subagent registration (separate file to
                           break the circular import with engine)
  memory.py              Memory class (history list + system_prompt field)
  prompts.py             build_system_prompt(cfg, is_subagent), preset
                           registry, env / git substitutions
  llm.py                 OpenAI-compatible streaming, run_assistant_turn(memory, ...)
  tools.py               ToolDef registry + 6 built-ins
  trace.py               TraceWriter (thread-safe), make_event
  config.py              RunConfig dataclass (DEFAULT_CONFIG derived)

SWE-BENCH LAYER
  swebench_dispenser.py  Workload, characterize, select, build_workload,
                           reset_repo, capture_diff, write_predictions

VENDOR
  vendor/providers.py    cheetahclaws' multi-backend streaming
                           (vendored, unused — reserved for vLLM)
```

**Trace dir layout:** one folder per root episode. `traces/episode_<root_id>/` holds the root `.jsonl`, all subagent `.jsonl` siblings, plus dispenser-emitted `.diff` / `.predictions.json` and analyzer-emitted `.analysis.txt`.

### `engine.py` (~430 lines)

Main agentic loop. Owns:

- `Episode` (frozen dataclass) — identity: `episode_id`, `run_id`, `depth`, `root_episode_id`, `parent_*`. Classmethods `new_root()` and `new_child(parent, parent_step_id)`. `.to_event_fields()` returns the seven identity fields as a dict for spreading into `make_event` calls.
- `Reservation` (frozen dataclass) — `(ok: bool, reason: str)`. Returned by `SubagentBudget.reserve`.
- `SubagentBudget` — thread-safe (one `threading.Lock`). Holds `_total`, `_per_parent: dict[str, int]`, `_concurrent` counters. `reserve(parent_episode_id, parent_depth) → Reservation`; `release()`. The `_total` and `_per_parent` counters are **monotonic** (they never decrement — they're lifetime caps); only `_concurrent` decrements on release.
- `RuntimeContext` (frozen dataclass) — per-tool-call: `(episode, writer, budget, step_id, tool_call_id, cfg)`. Constructed fresh in `ToolExecutor._run_one` for every tool invocation. **This is the only runtime state a tool function sees.**
- `ToolResult` (frozen dataclass) — return shape from the executor: `(tool_call_id, name, output, started_at_ms, ended_at_ms, latency_ms, exit_status)`.
- `ToolExecutor` — wraps tool dispatch. Holds `(cfg, writer, episode, budget, _cfg_dict)`. `execute(tool_calls, step_id)` picks `_execute_serial` or `_execute_parallel` via `_can_parallelize`. Parallel uses a fresh `ThreadPoolExecutor` per step (context-manager scope; threads don't persist across steps). `_tool_is_concurrent_safe(tool_def)` consults `cfg.subagent_parallelism` as a special case for `spawn_subagent`.
- `_validate_subagent_parallelism(cfg)` — called at `run_episode` entry. Raises `NotImplementedError` for future modes, `ValueError` for typos.
- `_filter_schemas(schemas, cfg)` — removes `spawn_subagent` from LLM-visible schemas when `enable_subagents=False`.
- `_resolve_trace_path(trace_path, cfg, episode)` — when no explicit path is given, returns `<output_dir>/<root_episode_id>/<episode_id>.jsonl`. Folder name uses `root_episode_id`; filename uses `episode_id`. For root, those are equal; for children, they differ — both land in the same per-root-episode dir.
- `run_episode(config, trace_path, task_input, episode=None, budget=None, workload_info=None)` — the step loop. `episode` and `budget` defaults let subagents reuse the function recursively. `workload_info` (any dict, default `None`) is merged into the `episode_start` event for dispenser-driven runs. Built once at episode start: resolves the system prompt (`prompts.build_system_prompt(cfg, is_subagent=episode.depth > 0)`), constructs `Memory.with_initial_user(task, system_prompt=...)`, then writes `episode_start` as the very first trace event.

Module-level state: `SUPPORTED_SUBAGENT_PARALLELISM`, `FUTURE_SUBAGENT_PARALLELISM` (mode-string sets).

### `memory.py` (~45 lines)

`Memory` dataclass — owns the conversational history for one episode. Provider-agnostic; no OpenAI knowledge.

- Fields: `messages: list[dict[str, Any]]`, `system_prompt: str`.
- `Memory.with_initial_user(content, system_prompt)` — episode-start constructor.
- `append_assistant(text, tool_calls)` / `append_tool(tool_call_id, name, content)` — the two write paths used by `run_episode`.
- `final_assistant_text()` — reverse-scans for the last assistant message with no tool_calls; used as the subagent's return value (replaces the old `engine._final_assistant_text` helper).
- `__len__` — so `len(memory)` works in `history_length` trace fields.

### `prompts.py` (~100 lines)

System-prompt construction. Inputs: `cfg: RunConfig`, `is_subagent: bool`. Output: a `str`.

- `MINIMAL` — a static 4-line prompt; the subagent default.
- `AGENT_TEMPLATE` — multi-section template with `{date}`, `{cwd}`, `{platform}`, `{git_info}`, `{parallel_tools_block}`, `{subagents_block}` slots.
- `PARALLEL_TOOLS_BLOCK`, `SUBAGENTS_BLOCK` — text fragments injected when the corresponding cfg flag is on. Empty otherwise.
- `REGISTRY: dict[str, str]` — preset name → template. Currently `"minimal"` and `"agent"`.
- `build_system_prompt(cfg, is_subagent=False)` — picks `"minimal"` if subagent, else `cfg.system_prompt`. If the name is in REGISTRY, runs `.format()` with substitutions; otherwise treats `cfg.system_prompt` as a raw override (no format, no crash on stray braces). Then appends `cfg.append_system_prompt` if non-empty.
- `_git_info()` — runs `git rev-parse --abbrev-ref HEAD` with a 2-second timeout, returns `"- Git branch: <name>\n"` or `""` on any failure (non-repo, missing git, timeout).

Built once at episode start (engine stores the result on `Memory.system_prompt`), so git only runs once per episode.

### `llm.py` (~270 lines)

OpenAI-compatible streaming. Imports `Memory` from `memory.py`; signature changed in the memory.py refactor.

Public surface:

- `AssistantTurn` dataclass — `(text, tool_calls, input_tokens, output_tokens, latency_ms, ttft_ms, prefill_time_ms, decode_time_ms, measurement, finish_reason)`.
- `run_assistant_turn(memory: Memory, tool_schemas, config) -> AssistantTurn` — one LLM call, streaming. Reads `memory.messages` and `memory.system_prompt`.
- `detect_provider(model)` — prefix-based provider detection (`gpt-`, `o1`, `o3`, `o4` → openai; `provider/model` slash syntax → explicit provider).
- `messages_to_openai(messages)` — neutral → OpenAI format converter.
- `tools_to_openai(tool_schemas)` — reshape `{name, description, input_schema}` → `{type: function, function: {name, description, parameters}}`.
- `_build_openai_messages(memory)` — internal: prepends `{"role": "system", "content": memory.system_prompt}` if non-empty, runs `messages_to_openai` on the result.

Timing capture: `request_start` at the top of the call, `first_token_at` on the first delta with content or tool_calls, `request_end` after the stream ends. `ttft_ms = first_token_at - request_start`, `decode_time_ms = request_end - first_token_at`, `latency_ms = request_end - request_start`. `prefill_time_ms` is set equal to `ttft_ms` today — when we add a real prefill measurement (requires server-side data), that alignment can break. The `SYSTEM_PROMPT` constant that used to live here was deleted; prompts.py owns prompt content now.

Module-level state: `PROVIDERS` dict (openai, custom), `_PREFIXES` tuple for detection.

### `tools.py` (~440 lines)

Tool registry + 6 tools: `bash`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`. Every tool func has signature `func(params, config, context) -> str`; `context` is `RuntimeContext | None` and is unused by all tools except `spawn_subagent`.

Public surface:

- `ToolDef` (dataclass, not frozen — allows runtime toggling of flags if ever needed): `(name, schema, func, read_only, concurrent_safe)`.
- `register_tool(tool_def)`, `get_tool(name)`, `get_all_tools()`, `get_tool_schemas()`, `clear_registry()`.
- `execute_tool(name, params, config=None, context=None) -> str` — registry dispatch with max-output truncation (`max_tool_output` = 32000 chars default; middle is elided with a `[... N chars truncated ...]` marker). On a raised exception, returns `f"Error: executing {name}: {exc}"` — the leading `"Error:"` matches engine's `exit_status` classifier (B1 fix).

Per-tool `concurrent_safe` values:

| Tool | read_only | concurrent_safe | Reasoning |
|---|---|---|---|
| `read_file` | True | True | Pure read |
| `glob` | True | True | Directory walk |
| `grep` | True | True | Subprocess or read, no shared state |
| `bash` | False | False | Shared cwd + arbitrary side effects |
| `write_file` | False | False | Can race with reads/writes on same path |
| `edit_file` | False | False | Same |
| `spawn_subagent` (from `subagent.py`) | False | False static; toggled at runtime by `ToolExecutor._tool_is_concurrent_safe` based on `cfg.subagent_parallelism` |

Module-level state: `_registry: dict[str, ToolDef]`. `_register_builtins()` runs once at import, populating the registry. Tests that load `tools.py` via `importlib.util.spec_from_file_location` get a fresh module instance with a fresh registry — be aware if you construct tests with `_load_module`.

### `subagent.py` (~155 lines)

Owns `AgentDefinition` (schema inspired by cheetahclaws' multi_agent layer) and `_spawn_subagent`. Exists **specifically to break the circular import** that would arise if `spawn_subagent` lived in `tools.py`: its func does `from engine import Episode, run_episode` **inside the function body**, not at module top. This is load-bearing — moving it to the top works *today* because engine doesn't import subagent, but makes the file fragile to future refactors.

`runner.py` and `profile_swebench.py` both import `subagent` (as `import subagent  # noqa: F401`) for the registration side-effect. Tests that don't exercise subagents skip this import; the tool simply isn't in the registry for those tests, which is fine.

`_spawn_subagent` order of operations:

1. Check `context` is not None → else return error
2. Check `config["enable_subagents"]` is true → else return error (defense in depth; schemas are filtered too)
3. Check `task` param is non-empty → else return error
4. `budget.reserve(parent_episode_id, parent_depth)` → return error string if rejected
5. Build `child_config` (copy of dict, apply `model` / `max_steps` overrides from params)
6. Construct `child_episode` via `Episode.new_child(parent, parent_step_id=context.step_id)`
7. Compute `child_trace_path = output_dir / child_episode.root_episode_id / f"{child_episode.episode_id}.jsonl"` — root_episode_id is inherited from the parent, so children always land as siblings of the parent's trace.
8. Write `subagent_start` event to the **parent's** trace writer
9. Call `run_episode(config=child_config, trace_path=child_trace_path, task_input=task, episode=child_episode, budget=budget)` — inside a `try/finally`
10. `budget.release()` in `finally`
11. Write `subagent_end` event to the parent's trace with `duration_ms` and child status
12. Return `result["final_text"]` (or `"(subagent produced no final text)"` if empty)

### `trace.py` (49 lines)

Minimal. `make_event(event_type, step_id=None, **fields)` is just a dict builder with `timestamp` + `type` + `step_id` baked in; everything else comes from `**fields`. Engine passes episode linkage via `**episode.to_event_fields()`.

`TraceWriter`:

- `__init__(path)` — creates parent dir (including the per-root-episode subdir), opens file in append mode (`"a"`), initializes `_lock = threading.Lock()`.
- `write(event)` — acquires lock, `json.dump`, newline, `flush`. Lock guards the full write+newline+flush so concurrent events don't interleave mid-line.
- `close()` — closes file handle if not already closed.
- Supports context-manager protocol (`__enter__` / `__exit__` calls close).

**The lock is load-bearing** for parallel execution. Removing it will cause interleaved JSONL garbage under `allow_parallel_tools=True`.

### `config.py` (~62 lines)

`RunConfig` (frozen dataclass) is the **single source of truth** for all 20 knobs and their defaults. `DEFAULT_CONFIG: dict[str, Any] = asdict(RunConfig())` derives the dict from it at module load — adding a knob is one dataclass field, the dict syncs automatically.

- `RunConfig.from_dict(data)` filters unknown keys via `dataclasses.fields(cls)` so old config files don't crash. `to_dict()` uses `asdict`.
- `load_config(path=None)` returns a dict (defaults merged with any JSON file contents). `save_config(config, path)` writes pretty-printed JSON.

The dict↔dataclass seam is the boundary: `engine.py` works with `RunConfig`; `llm.py`, `tools.py`, and the dispenser take dicts.

### `swebench_dispenser.py` (~165 lines)

The SWE-bench library — a pure-Python layer above the engine, no LLM imports.

- `InstanceCharacteristics` (frozen dataclass) — `instance_id`, `repo`, `problem_statement_length`, `hints_present`, `num_fail_to_pass`, `num_pass_to_pass`, `base_commit`, `created_at`.
- `Workload` (frozen dataclass) — `instance_id`, `repo_path`, `task_input`, `characteristics`, `raw_instance`.
- `load_instances(jsonl_path)` — reads the SWE-bench JSONL, returns `list[dict]`.
- `characterize(instance)` — converts one raw instance dict to `InstanceCharacteristics` (parses the JSON-encoded `FAIL_TO_PASS` / `PASS_TO_PASS` strings, treats blank `hints_text` as absent).
- `select(instances, *, min_ps_len, max_ps_len, repo, instance_id, limit)` — filter helper.
- `build_workload(instance, repos_dir)` — verifies the per-instance clone exists at `<repos_dir>/<instance_id>/` and that `base_commit` resolves in its history (`git cat-file -e <sha>^{commit}`); returns a `Workload`.
- `reset_repo(repo_path, base_commit)` — `git reset --hard <base_commit> && git clean -fd`. Idempotent; defends against agent-staged or agent-committed state.
- `capture_diff(repo_path, base_commit)` — stages everything (`git add -A`) then `git diff --cached <base_commit>` so untracked files are in the diff too. Cumulative against base, regardless of whether the agent committed mid-run.
- `render_task(instance, repo_path)` — `problem_statement + repo pointer`. The "stop hint" was dropped; the `agent` system prompt covers termination.
- `write_predictions(instance_id, model_patch, output_path)` — emits a SWE-bench-format predictions file the eval harness can consume later.

### `profile_swebench.py` (~150 lines)

The SWE-bench CLI driver. Iterates selected instances, runs each through `engine.run_episode` with `workload_info=asdict(characteristics)`, captures the diff and predictions next to the trace.

- `_build_config(args)` — turns CLI args into a config dict (`--system-prompt`, `--output-dir`, `--config <json>`). Also pre-resolves `output_dir` to absolute against the script-start cwd, *before* the per-instance chdir loop. This is the B3 fix: without pre-resolution, traces would land inside the SWE-bench repo (and `capture_diff`'s `git add -A` would sweep them up).
- `_run_one(instance, repos_dir, cfg_overrides, original_cwd)` — three-phase: setup (`build_workload` + `reset_repo`), run (`chdir` → `run_episode` → `chdir` back via `try/finally`), capture (`capture_diff` → write `.diff` and `.predictions.json` next to the trace). Returns `(success, summary_line)`. Setup failures return `status=skipped`; run failures return `status=crashed`; capture failures degrade to `warning=capture_failed:...` but still count as success.
- `main()` — exits 0 if every instance succeeded, else 1.

### `analyzer.py` (~310 lines)

The trace analyzer. Single-file input; output is a per-episode summary plus (when applicable) a recursive subagent tree, full `cfg`, and `workload_info`.

- `EpisodeSummary` (dataclass) — derived metrics: `total_wall_ms` (from episode_start/end timestamps), `total_llm_ms` (sum of `llm_call.latency_ms`), `total_tool_ms` (per-step `max(ended_at_ms) - min(started_at_ms)` summed — handles parallel correctly without over-counting), `prompt/completion_tokens_total`, `tool_call_count_by_name`, `subagent_count`, plus identity fields and the full `cfg` / `workload_info` from `episode_start`.
- `summarize(rows)` — builds an `EpisodeSummary` from a parsed JSONL list. Raises on missing `episode_start` or `episode_end`.
- `render_summary(summary)`, `render_workload(workload)`, `render_config(cfg)` — text renderers.
- `TreeNode` + `build_tree(trace_path)` — recursively follows `subagent_end.child_trace_path`. Relative paths resolve via *basename* in the trace's parent dir (works for both the new per-root-episode layout and the legacy flat layout). Missing children render as `(MISSING: trace file not found)` rather than crashing.
- `render_tree(node)` — indented ASCII tree with box-drawing connectors (`├─`, `└─`, `│  `, `   `). Each node line: `episode_id (depth N, spawned at step M, status, K steps)`.
- `main(argv)` — single trace path arg. Prints summary → workload → config → tree. Auto-saves the same to `<trace_dir>/<trace_stem>.analysis.txt`.

### `runner.py` (~55 lines)

Plain CLI for one task → one episode. Loops `runs_per_task` times serially (per-run + aggregate wall time printed). The only opinion it imposes vs `DEFAULT_CONFIG` is `allow_parallel_tools=True` — kept as a runner-specific choice.

### `run_swebench.sh` (project root, executable)

End-to-end shell wrapper: dispense → profile → analyze each trace produced. Forwards filter / config args to `profile_swebench.py`. Captures dispenser stdout via `tee`, greps for `trace=<path>`, runs `analyzer.py` on each. Propagates the dispenser's exit code via `PIPESTATUS`.

## 3. Code Provenance

Crucial for future sessions: knowing what came from `cheetahclaws` vs. what's ours vs. what's modified.

### Vendored verbatim (unused today)

| File | Size | Source | Status |
|---|---|---|---|
| `vendor/providers.py` | 696 lines | `cheetahclaws/providers.py` | Byte-for-byte copy. Unused by current runtime. Reserved for vLLM step. |

### Lifted with modifications

| Piece | Source | Our version | What changed |
|---|---|---|---|
| Tool impl `_read_file` | `cheetahclaws/tools.py::_read` | `tools.py::_read_file` | Removed `newline=""` kwarg (Python 3.13+ only); same logic otherwise |
| Tool impl `_write_file` | `cheetahclaws/tools.py::_write` | `tools.py::_write_file` | Same kwarg removal; renamed to `_write_file` |
| Tool impl `_edit_file` | `cheetahclaws/tools.py::_edit` | `tools.py::_edit_file` | Same kwarg removal; renamed |
| Tool impl `_bash` | `cheetahclaws/tools.py::_bash` | `tools.py::_bash` | Identical |
| Tool impl `_glob_tool` | `cheetahclaws/tools.py::_glob` | `tools.py::_glob_tool` | Renamed to avoid shadowing `glob` module; identical logic |
| Tool impl `_grep_tool` | `cheetahclaws/tools.py::_grep` | `tools.py::_grep_tool` | **Fixed bug**: cheetahclaws passes `--no-heading` to plain `grep` (it's a ripgrep-only flag). Our fallback uses `-r` + no `--no-heading`. |
| Tool impl `_kill_proc_tree` | `cheetahclaws/tools.py::_kill_proc_tree` | `tools.py::_kill_proc_tree` | Identical |
| Diff helpers (`generate_unified_diff`, `maybe_truncate_diff`) | `cheetahclaws/tools.py` | `tools.py::_unified_diff`, `_truncate_diff` | Renamed, unchanged |
| `ToolDef` dataclass shape | `cheetahclaws/tool_registry.py::ToolDef` | `tools.py::ToolDef` | Same fields `(name, schema, func, read_only, concurrent_safe)`; our `func` signature widened to `(params, config, context)` |
| Registry pattern | `cheetahclaws/tool_registry.py` | `tools.py` | Same pattern: module-level `_registry` dict, `register_tool`, `get_tool`, `get_tool_schemas`, `clear_registry`. `execute_tool` accepts `context=None` (ours adds this param). |
| `tools_to_openai` schema converter | `cheetahclaws/providers.py::tools_to_openai` | `llm.py::tools_to_openai` | Identical logic |
| `messages_to_openai` converter | `cheetahclaws/providers.py::messages_to_openai` | `llm.py::messages_to_openai` | Same shape; ours is simpler (only openai backend) |
| Streamed tool-call reconstruction | `cheetahclaws/providers.py::stream_openai_compat` | `llm.py::run_assistant_turn` | Same pattern: buffer `tool_calls[index]` across chunks, assemble at end |
| `build_system_prompt` pattern | `cheetahclaws/context.py::build_system_prompt` | `prompts.py::build_system_prompt` | Same shape (template constant + `.format()` substitutions for env / git info, conditional appended sections). Our template body is ours; the assembly pattern is the lift. We added a registry of named presets and a raw-override path that bypasses formatting. |
| Neutral message format | `cheetahclaws/agent.py::AgentState.messages` | `memory.py::Memory.messages` | Same shape: user / assistant-with-tool_calls / tool roles. Cheetahclaws keeps history on a global agent state; ours lives on a per-episode `Memory` dataclass owned by `run_episode`. |
| `AgentDefinition` idea | `cheetahclaws/multi_agent/subagent.py::AgentDefinition` | `subagent.py::AgentDefinition` | Trimmed: `(name, description, system_prompt?, model?, max_steps?)` only. No `tools`, no `source` |

### Written from scratch

| File / component | Notes |
|---|---|
| `engine.py` entire file | Our control layer. `run_episode`, `Episode`, `Reservation`, `SubagentBudget`, `RuntimeContext`, `ToolResult`, `ToolExecutor`, all helpers. |
| `trace.py` entire file | Minimal JSONL writer. Thread-safe. No cheetahclaws equivalent. |
| `memory.py` entire file | Per-episode `Memory` dataclass. Pure state — no provider knowledge. Append helpers replace the old engine-level `_final_assistant_text` and inline dict appends. |
| `prompts.py` template content | Template body, registry, and the parallel/subagent conditional blocks are ours. The `build_system_prompt(cfg, is_subagent)` pattern itself is the lift (above). |
| `config.py::RunConfig` dataclass | Our typed representation. `DEFAULT_CONFIG` dict fields partially overlap with cheetahclaws config but we diverge in naming and semantics (especially subagent knobs). After consolidation, `DEFAULT_CONFIG = asdict(RunConfig())` — RunConfig is the single source of truth. |
| `config.py::load_config` / `save_config` | Simple JSON file I/O. |
| `runner.py` | Ours. ~55 lines. |
| `subagent.py::_spawn_subagent` | Our design. Recursion + budget reserve/release + `subagent_start` / `subagent_end` events + per-root-episode `child_trace_path` computation. Cheetahclaws' subagent uses a thread pool + queue; we don't. |
| `ToolExecutor` class | Our design. Single abstraction for serial + parallel tool dispatch, with Option A safety gate. |
| `SubagentBudget` class | Our design. Four-cap budget with one reservation/release interface. |
| `Episode` dataclass with parent linkage | Our design. |
| `RuntimeContext` | Our design. Cheetahclaws has a `runtime.py` singleton which we explicitly chose not to use. |
| Tool lambdas + registration order in `tools.py::_register_builtins` | Ours. Cheetahclaws registers tools differently (schema-first with TOOL_SCHEMAS list). |
| Trace event types & field names (incl. `episode_start` with full cfg + workload_info) | Ours. |
| `swebench_dispenser.py` entire file | Our design (per `workload.md`). Per-instance dir convention, characterize / select / build / reset / capture / write_predictions. No cheetahclaws equivalent. |
| `profile_swebench.py` entire file | Our SWE-bench CLI driver. Sets up `workload_info` for `run_episode`, manages `chdir` per instance, post-run capture. |
| `analyzer.py` entire file | Our trace analyzer. Per-episode metrics with parallel-aware tool wall calc, recursive subagent tree across files, full cfg + workload dump, auto-saved `.analysis.txt`. |
| `run_swebench.sh` | Our end-to-end wrapper at the project root. |

### Explicitly not lifted

From cheetahclaws, we deliberately did **not** bring in:

- `runtime.py` (singleton globals — forbidden in our design, §11)
- `multi_agent/SubAgentManager` + `multi_agent/tools.py` (too heavy; we use recursion, D7)
- `compaction.py` (history compression — deferred to future knob)
- `permissions` / `plan_mode` / `bridges` / `mcp/` / `voice/` / `video/` / `ui/` / `cloudsave.py` / `checkpoint/` / `memory/` / `skill/` / `task/` / `plugin/`
- `AskUserQuestion`, `SleepTimer`, `NotebookEdit`, `GetDiagnostics`, `WebFetch`, `WebSearch` tools — not needed for headless profiling
- All their testing infrastructure (they use a different harness)

From claude-code (TypeScript, `/home/user0/nanoLens/claude-code/`), we didn't import any code — only the design inspiration for the sidechain-JSONL-per-agent pattern and the optimistic-staleness concurrency model (D8). Everything in ClawTrace is Python.

## 4. Decisions We Weighed

### D1. Vendoring vs importing cheetahclaws

**Chose: vendor.** Copy `providers.py` into `clawTrace/vendor/`, pin the version. Reasoning: profiling runs must be reproducible — silent upstream changes would confound results. Downside: bug fixes don't propagate automatically. Worth it for this use case.

### D2. Replace `llm.py` with `providers.stream()` now, or defer?

**Chose: defer.** Keep our 270-line openai-specific `llm.py` until we actually need vLLM. `providers.stream()` covers 10 backends we don't use; adopting it now would add machinery we can't verify against real workloads. The swap happens when vLLM lands — that's when the abstraction pays off.

### D3. Lift more tools or stay minimal?

**Chose: lift 4 more** (`write_file`, `edit_file`, `glob`, `grep`), stop before network/specialized ones (`WebFetch`, `NotebookEdit`, `GetDiagnostics`). These six are what agentic workloads (SWE-bench etc.) actually need. Network tools are flaky for profiling.

### D4. History compaction now or later?

**Chose: later.** `history_policy` is a future knob. Compaction adds its own LLM call path that complicates profiling interpretation. Baseline first.

### D5. `AgentDefinition`: reuse cheetahclaws' schema or write our own?

**Chose: lift it.** Low-risk reuse. Matches the `spawn_subagent` tool's input schema naturally.

### D6. Parallel tool calls — Option A vs Option B gate

Option A: parallel only if **all** tools in the batch are `concurrent_safe`. Otherwise fall back to serial.
Option B: partition each turn into a concurrent-safe group + a serial group.

**Chose: A.** One-line rule. Bash is not concurrent-safe (shared cwd, races), so mixed turns serialize. Option B would need more trace event shape and harder analysis for little throughput win in typical workloads.

### D7. Subagent mechanism — thread pool, subprocess, async, or recursion?

Surveyed cheetahclaws (`ThreadPoolExecutor` + `SubAgentManager`) and claude-code (`void` fire-and-forget async + sidechain JSONL + per-agent AbortController).

**Chose: plain recursion, piggyback on `ToolExecutor`.**

- `spawn_subagent` is a regular registered tool.
- Its func calls `run_episode(..., parent=ctx.episode, budget=ctx.budget)` recursively.
- Parallelism comes from `ToolExecutor`'s thread pool, not from a separate subagent pool.
- Recursion is **vertical** (depth, via Python call stack); threads are **horizontal** (width, via `ThreadPoolExecutor`). They compose.

Rejected alternatives:

| Mechanism | Why not |
|---|---|
| Dedicated subagent `ThreadPoolExecutor` (cheetahclaws pattern) | Redundant. `ToolExecutor` is already a thread pool. |
| Async `void` promises (claude-code pattern) | Would require rewriting `run_episode` as `async def` + tool impls. Big lift for a capability we don't use (no UI streaming). |
| Subprocess isolation | Process-start overhead confounds profiling measurements. |
| Git worktree isolation | ClawTrace doesn't mutate the workspace during profiling. Deferred as a future `subagent_parallelism="worktree"` mode. |
| Inter-agent inbox queues | No use case. Subagents are one-shot. |

### D8. Subagent parallelism — the write-conflict problem

**The hole we caught:** marking `spawn_subagent` with `concurrent_safe=True` lets parallel subagents race on `bash`, `write_file`, and `edit_file` inside the children. `concurrent_safe` describes the tool's *own* execution; a subagent delegates to arbitrary child work.

Surveyed how claude-code handles this: **optimistic staleness detection** via per-agent `FileStateCache` + mtime checks in Write/Edit. Elegant, but doesn't guard `bash`, and adds ~40 lines of plumbing we don't need today.

**Chose: make it a config mode.** `subagent_parallelism` has four values:

| Value | Status | Semantics |
|---|---|---|
| `"serial"` | shipping, default | `spawn_subagent` treated as not concurrent-safe → parallel batches fall through to serial. Safe under all tools. |
| `"shared"` | shipping | `spawn_subagent` treated as concurrent-safe. User asserts no write conflicts. `config_warning` trace event is emitted. |
| `"shared+optimistic"` | future | Will add claude-code's mtime check. Currently raises `NotImplementedError`. |
| `"worktree"` | future | Will add cheetahclaws-style git worktree isolation. Currently raises `NotImplementedError`. |

Unknown modes raise `ValueError` at episode start so typos fail loudly.

### D9. Trace file layout for subagents

**Chose: one JSONL file per episode, with parent linkage in every event.**

- Each subagent has its own trace file.
- Every event in every trace carries `root_episode_id` + `depth` + `parent_*` fields so post-hoc analysis can reconstruct full trees.
- Parent trace carries `subagent_start` + `subagent_end` pairs tagged with `child_episode_id` + `child_trace_path`.

Alternative was a single nested file; we chose per-episode to keep the existing invariant ("one episode = one file") and because any single child file is still self-describing.

### D10. `subagent_strategy` knob

**Chose: drop it for now.** No second strategy to name, so the knob would have nothing to discriminate between. Re-add when there's a concrete second option.

### D11. `RunConfig` dataclass vs dict

**Chose: both.** Dict at the boundary (disk JSON, public `run_episode(config=...)`), `RunConfig` internally. Boundary stays simple; internal code gets typo-safety and IDE hints.

### D12. Where do new runtime pieces live?

**Chose: keep flat until forced to split.**

- `SubagentBudget`, `RuntimeContext`, `ToolExecutor`, `Episode` all live in `engine.py`. Will extract `ToolExecutor` to `executor.py` if `engine.py` crosses ~500 lines.
- `subagent.py` was forced: it needs to import `engine.run_episode` for recursion, so keeping `spawn_subagent` in `tools.py` would create a cycle.
- No `events.py`, no `types.py`, no packaging.

## 5. The Mental Model

### Step loop

1. `step_start` event
2. `run_assistant_turn(history, tool_schemas, cfg)` → `AssistantTurn`
3. `llm_call` event
4. Append assistant message to history
5. `ToolExecutor.execute(tool_calls, step_id)` → list of `ToolResult`
6. Append tool results to history
7. `context_update` event
8. `step_end` event
9. Stop if no tool calls OR `step_id + 1 == max_steps`

### Parallelism, compactly

- One concurrency abstraction: `ToolExecutor` = a `ThreadPoolExecutor` with a safety gate.
- Gate (`_can_parallelize`): `cfg.allow_parallel_tools` AND batch size ≥ 2 AND every tool is concurrent-safe.
- `spawn_subagent`'s concurrent-safety is config-dependent: `_tool_is_concurrent_safe` consults `cfg.subagent_parallelism`.
- Parallel subagents therefore need TWO flags: `allow_parallel_tools=True` AND `subagent_parallelism="shared"` (or a future non-serial mode).

### Trace invariants

- One `.jsonl` per episode.
- Every event has `episode_id`, `run_id`, `depth`, `root_episode_id`.
- Every event except root's has `parent_episode_id` + `parent_run_id` + `parent_step_id` (None at root).
- Tool timing fields are relative to the start of that step's tool-execution phase.

## 6. Concurrency Model Details

Worth a dedicated section because the nested parallelism has a few non-obvious properties.

### Thread topology

`ToolExecutor._execute_parallel` creates a **fresh** `ThreadPoolExecutor` per step (via `with` block), with `max_workers = min(len(batch), cfg.max_parallel_tools)`. When the block exits, threads are joined and freed. Threads do not persist between steps.

If `spawn_subagent` is called and runs in a worker thread T1, the subagent's `run_episode` runs inside T1. The subagent's own `ToolExecutor` creates a fresh pool *inside* T1 when it needs parallelism. So nested parallelism looks like:

```
Main thread
  └─ step N ─ ToolExecutor.pool (workers: 1..max_parallel_tools)
       └─ T1 (running spawn_subagent body in run_episode)
            └─ child step M ─ ToolExecutor.pool (child's own, workers: 1..max_parallel_tools)
                 └─ T1.1, T1.2, ... (grandchildren tool calls)
```

`ThreadPoolExecutor` nesting is supported in Python. Thread pools don't share workers across nesting levels, so total thread count grows as `max_parallel_tools ** depth` in the worst case. With defaults `max_parallel_tools=4, max_subagent_depth=0` (unlimited), pathological workloads could exhaust OS thread limits. In practice, set `max_subagent_depth` to a small number when enabling subagents.

### Lock scope

One lock: `TraceWriter._lock` (a `threading.Lock`). Scope: the full `json.dump + write("\n") + flush` sequence. Nothing else in the project takes locks across thread boundaries.

`SubagentBudget` has its own `threading.Lock` — scope: the `reserve()` and `release()` critical sections. These two locks never interact: `TraceWriter` writes don't touch budget state, budget operations don't write to the trace.

### What isn't locked

- `_registry` (module-level dict in `tools.py`) is **not** locked. It's populated once at import and then read-only. If you ever add runtime `register_tool` calls from worker threads, add a lock.
- `history: list[dict]` in `run_episode` is **not** locked. Tool results are appended *after* the parallel batch completes (in the main thread), so concurrent append isn't a concern.
- `cfg.to_dict()` is called once per `ToolExecutor.__init__` and cached in `_cfg_dict`. Workers share this dict by reference. It's never mutated.

### The `max_concurrent_subagents` trap

With `max_concurrent_subagents=1` and `max_subagent_depth>1`, a subagent **cannot spawn grandchildren**. Reason: the parent reserved the one concurrent slot; the subagent itself is "in flight"; when the subagent tries to spawn its own child, `_concurrent=1` already equals the cap → rejected.

If you want deep subagent nesting, set `max_concurrent_subagents >= max_subagent_depth`, or leave it at 0 (unlimited).

### Ordering guarantees

- Tool result order in `history` = order of tool_calls emitted by the model. `_execute_parallel` gathers via `[future.result() for future in futures]` where `futures` is built in submission order — so blocking happens in submission order and the list preserves emission order.
- Trace event order in the JSONL file = order of `writer.write` calls. Under parallel execution, tool_call events for the same step may interleave in completion order (not submission order). Use `started_at_ms` / `ended_at_ms` to reason about timing; don't rely on file line order within a step.
- `subagent_start` is written before `run_episode` is called; `subagent_end` after it returns. So child's full trace is bracketed by parent's start/end — but because the child writes to a different file, there's no direct interleaving to worry about.

## 7. Config Knobs (all 20)

Grouped by what they change:

### Core runtime

| Knob | Default | Effect |
|---|---|---|
| `backend` | `"openai"` | Which provider (only `openai` implemented; vLLM via `custom` planned) |
| `model` | `"gpt-4o-mini"` | Model name; routed via prefix detection |
| `max_steps` | `8` | Hard cap on assistant turns per episode |
| `max_tokens` | `512` | Per-turn completion token cap |
| `temperature` | `0.0` | Sampling temperature |
| `tool_timeout_s` | `30` | Per-tool wall-clock cap (enforced by tools themselves where applicable) |
| `timeout_s` | `300` | Reserved episode-level timeout (not yet enforced) |
| `output_dir` | `"traces"` | Where JSONL files land |
| `openai_base_url` | `""` | Override for custom OpenAI-compatible endpoints |

### Parallel tool execution

| Knob | Default | Effect |
|---|---|---|
| `allow_parallel_tools` | `False` | Enable the parallel gate at all |
| `max_parallel_tools` | `4` | Worker count cap when parallel is allowed |

### Subagents

| Knob | Default | Effect |
|---|---|---|
| `enable_subagents` | `True` | Filter `spawn_subagent` out of LLM-visible schemas when disabled |
| `max_subagents_total` | `2` | Lifetime cap across the whole root tree (`0` = unlimited) |
| `max_subagents_per_parent` | `2` | Lifetime cap per individual parent (`0` = unlimited) |
| `max_subagent_depth` | `1` | Nesting depth cap (`0` = unlimited) |
| `max_concurrent_subagents` | `2` | How many may be in flight simultaneously |
| `subagent_parallelism` | `"serial"` | See D8 above |

### Runner

| Knob | Default | Effect |
|---|---|---|
| `runs_per_task` | `1` | Repeated-run factor; runner loops `run_episode` N times serially and prints per-run + aggregate timing |

### System prompt

| Knob | Default | Effect |
|---|---|---|
| `system_prompt` | `"agent"` | Registry name (`"agent"`, `"minimal"`) selects a preset from `prompts.py`; any other string is treated as a raw override and not formatted |
| `append_system_prompt` | `""` | Free-form text appended after the resolved prompt; applies to both main agent and subagents |

## 8. Trace Model

### Event types

- `episode_start` — first event in every trace; carries the run's identity and config snapshot (see fields below)
- `step_start` — beginning of a step (history snapshot)
- `llm_call` — one assistant turn's LLM metrics
- `tool_call` — one tool invocation, with timing offsets
- `subagent_start` — parent trace: spawn begin
- `subagent_end` — parent trace: spawn complete
- `context_update` — after tools: history growth, counts
- `step_end` — end of a step (stop_reason if any)
- `config_warning` — episode-start warning for risky config (e.g. `subagent_parallelism != "serial"`)
- `episode_end` — final summary

### Key field groupings

Every event carries `depth`, `root_episode_id`, and parent linkage via `episode.to_event_fields()`.

`episode_start` fields: `task_input`, `model`, `backend`, `system_prompt_name` (registry key or `"<custom>"`), `system_prompt_chars`, `cfg` (full snapshot), `workload_info` (dict | None — populated by the dispenser via the `workload_info` kwarg to `run_episode`, otherwise `null`). Subagent traces also get an `episode_start` (with `system_prompt_name="minimal"` and `workload_info=null`).

`llm_call` fields: `backend`, `model`, `latency_ms`, `ttft_ms`, `prefill_time_ms`, `decode_time_ms`, `prompt_tokens`, `completion_tokens`, `finish_reason`, `assistant_text_preview`, `tool_call_count`, `measurement`.

`tool_call` fields: `tool_call_id`, `tool_name`, `params`, `started_at_ms`, `ended_at_ms`, `latency_ms`, `exit_status`, `result_preview`.

**Note:** `step_start` no longer carries `task_input` — that information lives in `episode_start` (it doesn't change within an episode).

`subagent_end` fields: `child_episode_id`, `child_run_id`, `child_trace_path`, `child_status`, `child_stop_reason`, `child_step_count`, `duration_ms`, plus the `tool_call_id` that links to the corresponding `tool_call` event in the same step. On a crashed child, `child_status="crashed"`, `child_stop_reason="exception"`, `child_step_count=0`, and an `error` field carries the exception name + message. `subagent_end` is always written — either success or crash — so `subagent_start` events never orphan.

### Episode status values

`episode_end.status` is either:

- `"completed"` — the loop terminated because the assistant turn emitted no tool calls (`stop_reason="no_tool_calls"`). Natural end.
- `"incomplete"` — the loop terminated because `step_id + 1 == max_steps` (`stop_reason="max_steps_reached"`). The episode was cut off mid-task.

Consumers distinguishing "success" from "cut-off" should check `status` (not just `stop_reason`).

## 9. Failure Modes & Potential Errors

Enumerated so future debugging has a checklist. Most of these are low-probability but worth knowing.

### Registration / import

1. **Circular import between engine and subagent.** `subagent.py` imports `engine.run_episode` *inside* the function body of `_spawn_subagent`. If someone moves it to the module top *and* engine ever imports `subagent` (it doesn't today), you get an `ImportError` at cold start.
2. **Tool registration misses.** `subagent.py`'s registration happens on module import. If a test imports `engine` but not `subagent`, `spawn_subagent` won't be in the registry. `runner.py` includes the `import subagent` side-effect; tests must do the same explicitly (see `test_spawn_subagent_end_to_end`).
3. **Double-registration.** If `tools.py` is loaded twice (e.g. via `_load_module` test helper + normal `import tools`), you get two registry dicts — the one you see depends on which module object you're talking to. Tests using `_load_module` should use that module's `get_tool` consistently.

### Concurrency

4. **`max_concurrent_subagents=1` with `max_subagent_depth>1`**: subagents cannot spawn grandchildren. See §6 "The trap."
5. **OS thread exhaustion** under nested parallelism with unbounded depth. Set a depth cap when enabling subagents.
6. **`TraceWriter.close` during a write**: not currently racy because close is only called from `__exit__` after all worker futures have joined. If you ever add a cancellation path that closes mid-flight, add lock coverage to close.
7. **Tool func exceptions bypassing error-string conversion**: `execute_tool` wraps `tool.func` in try/except. If a tool raises, the result becomes `"Error executing {name}: {exc}"`. But a future that bypasses `execute_tool` (e.g. direct `tool.func` call in a custom executor) would propagate the exception via `future.result()`.

### Budget / subagents

8. **Budget reservation leaks.** `_spawn_subagent` uses `try/finally` around `run_episode` to always call `budget.release()`, including when `run_episode` raises. Any new code that calls `reserve` must mirror this. A leaked reservation keeps `_concurrent` high and blocks future spawns in the same tree.
9. **`_total` and `_per_parent` are monotonic.** They never decrement. That's intentional (lifetime caps), but means "concurrent cap reached" and "total cap reached" behave differently. A run hitting `max_subagents_total` can't recover.
10. **Parent writes in parallel subagents (`subagent_parallelism="shared"`)**: user-accepted risk. The `config_warning` event at episode start is the paper trail. If trace analysis shows mysterious file states, check for the warning first.
10a. **Subagent crash paths are traced.** If `run_episode` raises inside `_spawn_subagent`, the `finally` block releases the budget, and a `subagent_end` event is written with `child_status="crashed"` and an `error` field. `subagent_start` never orphans.

### Config / plumbing

11. **`RunConfig` vs dict confusion.** `engine.py` uses `RunConfig`; `llm.py` and `tools.py` use dicts. The seam is `cfg.to_dict()` at `ToolExecutor.__init__` and at the `run_assistant_turn` call site. Don't mix.
12. **Mutating `cfg_dict` in a tool.** `_cfg_dict` on `ToolExecutor` is cached once and shared across workers. If a tool mutates the config dict it receives, it affects subsequent tool calls in the same step and any parallel siblings. Our tools don't do this; don't introduce tools that do.
13. **`Episode.root_episode_id == episode_id` for root.** Code that assumes `root_episode_id != episode_id` to mean "not root" is wrong. Use `episode.depth == 0` for root detection.
14. **UUID collisions.** `episode_id` and `run_id` are 8-char hex (~4B possibilities each). Long-running profiling campaigns should still be safe, but if you generate 10⁵+ episodes in one run, bump to 16-char hex.
15. **`output_dir` shared across root + children.** All subagent trace files land in the same directory. To group a run, filter by `root_episode_id` in events.

### Streaming / LLM

16. **OpenAI delta format change.** `run_assistant_turn` accumulates `delta.tool_calls[index].function.arguments` across chunks. If OpenAI changes the streaming shape, reconstruction breaks silently (you'd see empty or malformed tool calls). `finish_reason="tool_calls"` is the early-warning signal.
17. **`prefill_time_ms = ttft_ms`.** We set them equal today. Code that treats them as independent will get the same value twice. Real prefill measurement requires server-side data; when added, align consumers.
18. **`max_tokens` cap on openai provider.** The code routes to `max_completion_tokens` for openai (required for o1/o3/o4/gpt-5) and `max_tokens` for other providers. A custom backend that doesn't accept `max_completion_tokens` would fail silently with a full-output completion.
19. **Empty `tool_calls` with `finish_reason="tool_calls"`**: shouldn't happen in practice but would appear as `tool_call_count=0` + odd finish_reason. Currently the engine just moves on (no tools to run → stop_reason="no_tool_calls" not set because we check `assistant_turn.tool_calls` not `finish_reason`). Worth a sanity check in analysis.

### Trace format

20. **Additive schema only.** New fields are safe; renames break downstream analysis silently. If a field must go, dual-write for a release.
21. **Field `step_id=None` on episode-scope events** (`episode_end`, `config_warning`) — intentional. Analysis code must handle `step_id=None`.
22. **`tool_call` event ordering under parallel execution.** Events appear in completion order, not submission order. Tools that finish faster write first. Use `started_at_ms` for submission-order reasoning, `ended_at_ms` for completion order.

### Python-version-specific

23. **`Path.read_text(newline="")` and `Path.write_text(newline="")`** are Python 3.13+. We stripped them for 3.12 compatibility. If re-added, 3.12 breaks.
24. **Frozen dataclasses with mutable fields** (`RunConfig`, `Episode`, `RuntimeContext` are all frozen) — can't mutate. Use `dataclasses.replace(instance, **changes)` if you need a modified copy.

## 10. What To Be Careful About

### Concurrent-safety is a promise about *this* tool's own execution

It does **not** transitively cover what a subagent's children do. That's why `spawn_subagent` is not concurrent-safe by default, and why `subagent_parallelism` exists as a separate mode knob.

### `TraceWriter` is thread-safe *only* because of the lock

If a future refactor drops the lock (or bypasses it), parallel trace events will interleave garbage in the JSONL. The lock is in `TraceWriter.write`; don't work around it.

### Import order is load-bearing

- `runner.py` imports `engine` first, then `subagent`.
- `subagent.py` has `from engine import run_episode` inside `_spawn_subagent` (not at module top) to avoid the cycle if anyone imports `subagent` before `engine`.
- If you add a new runtime module that registers tools, follow the same pattern: late-import `engine` inside the func.

### Trace event schemas are additive, not versioned

Add new fields freely (they become `null` for older analyzers). Don't rename or remove fields; downstream analysis scripts will break silently. If a field must go, introduce a new name, dual-write for a while, then cut over.

### `subagent_parallelism="shared"` is unsafe by the user's own assertion

If the workload writes files, parallel subagents can race. The `config_warning` event is the paper trail — don't suppress it. If you're analyzing a trace and can't explain file state, check for the warning first.

### Dict-based config at the boundary, `RunConfig` internally

Don't pass dicts into engine-internal functions, and don't pass `RunConfig` into `llm.py` / `tools.py` / tool funcs. The seam is at `run_episode`'s top and at `cfg.to_dict()` calls.

## 11. How To Scale

These are the rules we converged on during Steps 2–5. Follow them and new knobs drop in cleanly; violate them and `engine.py` becomes an `if cfg.get(...)` swamp.

### Rule 1 — Each knob has exactly one decision site

If `allow_parallel_tools` is read in two places, those two places will drift. It's read in `ToolExecutor._can_parallelize` and nowhere else.

Corollary: when adding a knob, decide where it's consulted and put a `TODO` everywhere else that looks like it should care. Better to centralize.

### Rule 2 — Knobs are parameters to existing abstractions, not new branches

- `allow_parallel_tools` isn't a new branch in the engine loop. It's an input to `ToolExecutor`, which already exists in every run.
- `subagent_parallelism` isn't a new execution mode. It's a gate inside `_tool_is_concurrent_safe`.
- A new subagent budget dimension isn't a new check scattered through the code. It's a new line in `SubagentBudget.reserve`.

### Rule 3 — One concurrency primitive

Don't add a second thread pool, async runtime, or subprocess fleet. `ToolExecutor`'s `ThreadPoolExecutor` is the one. Anything that needs parallelism inherits it (which is how `spawn_subagent` gets parallel subagents when configured).

### Rule 4 — Trace events are additive; no per-mode event type explosion

We added `config_warning`, `subagent_start`, `subagent_end` because they represent **new facts** (warnings, spawn events). We did **not** add `tool_call_parallel` or `tool_call_serial` because "serial or parallel" is a property, not a new kind of event. It goes in the `tool_call` event as timing offsets.

### Rule 5 — Keep `engine.py`'s loop readable as prose

If `run_episode`'s body stops reading like the 9-step list in section 4, the abstractions are leaking. Push complexity into `RunConfig` (definition), `ToolExecutor` (dispatch), `SubagentBudget` (budget rules), or `make_event` call sites (trace shape).

### Rule 6 — Split files only when forced

`subagent.py` exists because of a circular import. `engine.py` will cross 500 lines before we extract `ToolExecutor` to `executor.py`. Don't split by gut feeling.

## 12. Testing Strategy

### What's tested

- **Config** (3 tests): defaults, `RunConfig.from_dict` filtering, dict round-trip.
- **Engine** (15 tests): episode shape, multi-step events, history tracking, finish_reason independence, timing offsets, parallel gate (3 modes), parallel overlap timing, budget caps (depth, per-parent, concurrent), subagent end-to-end, error path when disabled, future-mode validation.
- **LLM** (3 tests): provider detection, message conversion, system-prompt builder.
- **Tools** (3 tests): registry, `read_file`, `bash`.

### What's not tested

- Live OpenAI calls (verified manually via `runner.py`).
- vLLM (not built).
- Real parallel write races (we handle them policy-wise, not technically).
- Trace analysis (no analysis script yet).

### Mocking strategy

`_with_fake_turns(turns, fn)` in `test_engine.py` monkey-patches `engine.run_assistant_turn` to pop from a list. This exercises the real engine loop, real tool dispatch, real trace writing — everything except the HTTP call.

For subagents, the fake-turns list must cover parent turns **and** child turns (popped in order).

## 13. Environment & Compatibility

- **Python**: developed against 3.12. Requires ≥3.10 (for PEP 604 `|` unions in type hints, used throughout). Frozen dataclasses need 3.7+.
- **Dependencies**: `openai` (Python SDK, ≥1.0 style — we use `client.chat.completions.create` with streaming). Tests additionally need `pytest`. No other runtime deps.
- **External tools expected**:
  - `ripgrep` (`rg`) — the `grep` tool prefers it. Falls back to plain `grep -r` if missing. Both Linux distros tested.
  - `git` — not required today, but will be needed for the future `subagent_parallelism="worktree"` mode.
- **OpenAI API**: `OPENAI_API_KEY` env var. Custom endpoints via `openai_base_url` config.
- **Platforms**: Linux (WSL tested). `_bash`'s `start_new_session=True` is POSIX-only; Windows falls back to non-group-killing. Signal handling in `_kill_proc_tree` branches on `sys.platform`.
- **Filesystem**: trace dir is created on demand (`TraceWriter.__init__`). No cleanup — old traces accumulate under `output_dir`. A future cleanup script can group by `root_episode_id`.
- **Clock**: we use `time.perf_counter()` for all timing (monotonic, suitable for durations). `datetime.now(timezone.utc)` for event timestamps (wall clock).

## 14. What We Discarded

- **Single-JSON-action runtime model** — original prototype; replaced by native tool-calling.
- **Mock runtime path** — removed; offline tests use fake turns instead.
- **`subagent_strategy` config knob** — dropped until a second strategy exists.
- **Dedicated subagent thread pool** — redundant with `ToolExecutor`.
- **Worktree / mtime isolation in Step 5** — deferred to named future modes of `subagent_parallelism`.
- **Mid-run cancellation of subagents** — can be bolted on with a shared `threading.Event` if needed.

## 15. What's Next

### Recently shipped

- **`analyzer.py`** — per-episode metrics (wall / llm / tool ms with parallel-aware tool wall calc, prompt+completion tokens, tool counts, subagent count), recursive subagent tree visualization (follows `subagent_end.child_trace_path` across files), full `cfg` + `workload_info` sections in the output. Auto-saves to `<trace_dir>/<trace_stem>.analysis.txt` next to the trace.
- **Per-root-episode trace dir layout** — `traces/episode_<root_id>/` is now one folder per run, holding the root `.jsonl`, all subagent `.jsonl` siblings, and (when applicable) `.diff` / `.predictions.json` / `.analysis.txt`. Engine's `_resolve_trace_path` and `subagent.py`'s `child_trace_path` both build this path.
- **`run_swebench.sh`** — end-to-end wrapper at the project root: dispense → profile → analyze each trace produced. Forwards filter / config args to `profile_swebench.py`.
- **`config.py` consolidation** — `DEFAULT_CONFIG` now derived from `RunConfig` via `asdict(RunConfig())`. Adding a knob is one dataclass field; the dict syncs automatically. Eliminates the silent-drift footgun the doc previously warned about.
- **Default values shifted toward "realistic agent"** — `enable_subagents=True`, subagent caps now `2/2/1/2` (per-parent / total / depth / concurrent) instead of unlimited / 1. `max_steps=20` (was `8`). `allow_parallel_tools` still defaults to `False`.
- **`memory.py`** — `Memory` class owns conversational history and system prompt; engine uses append helpers instead of poking dicts directly. `messages_to_openai` stayed in `llm.py` (provider-format converter belongs with provider transport).
- **`prompts.py`** — configurable system prompts with `"agent"` (multi-section template w/ env + git info + conditional parallel/subagent blocks) and `"minimal"` presets, raw-string overrides, `append_system_prompt`. Subagents always get `"minimal"` (hardcoded for now).
- **`episode_start` trace event** — first event in every trace, carries `task_input`, `model`, `backend`, `system_prompt_name`, `system_prompt_chars`, full `cfg` snapshot, and `workload_info`. Subsumed the dispenser's planned `workload_info` event; `step_start` no longer duplicates `task_input`.
- **`swebench_dispenser.py` + `profile_swebench.py`** — the SWE-bench layer. See the (now-built) design in [workload.md](workload.md). Per-instance repo convention `<repos_dir>/<instance_id>/`. Smoke-tested end-to-end against `astropy__astropy-12907`. Predictions file emitted alongside trace + diff for later eval (no Docker required).
- **B1 fix** (`tool_call.exit_status` now correctly classifies raised exceptions as `"error"`).
- **B3 fix** (dispenser pre-resolves `output_dir` to absolute before its per-instance chdir loop).

### Up next, in approximate priority order

1. **vLLM backend** — `vendor/providers.py` is waiting. Swap `llm.py` to use `providers.stream()`; gains ~10 backends including vLLM. Genuinely valuable now: dispenser can drive many SWE-bench instances against a self-hosted model cheaply, and analyzer can compare them.
2. **`history_policy`** — start with `"full"` (current) and `"snip_old_tool_results"` (lift from `cheetahclaws.compaction`). Memory class has the natural seam. The first real SWE-bench run produced 169K input tokens across 20 steps before hitting `max_steps`; unbounded history is now an observed cost, not just a theoretical one.
3. **Per-instance Python env setup for SWE-bench** — the astropy smoke run hit `max_steps_reached` partly because the agent kept trying to `bash` `pip install` / `python -c "import astropy"` and failing (the clawTrace venv has no per-instance deps). Either the dispenser pre-installs each instance's env, or we shell out to SWE-bench's harness Docker images. Without test feedback, agents edit blind. Higher leverage than (4) and (5) until fixed.
4. **TBT (time-between-tokens)** — record per-chunk arrival timestamps in `llm.py`, add to `llm_call` events.
5. **Non-serial `subagent_parallelism` modes** — `"shared+optimistic"` (~40 lines, mtime detection) and/or `"worktree"` (heavier).
6. **Optional `--eval` for `profile_swebench.py`** — predictions files are already emitted; running them through `swebench.harness.run_evaluation` is ~50 lines + Docker setup. Pairs naturally with (3) since both want the per-instance test environment.
7. **Analyzer extensions (deferred from v1.5)** — `--group-by`, `--filter`, `--json` output, directory input. Add once you have a real comparison to run.


## 16. How To Run

From `clawTrace/`:

```bash
source .venv/bin/activate
export OPENAI_API_KEY=sk-...
python runner.py "Read these three files and summarize each: README.md, implement.md, config.py"
```

Then inspect:

```bash
ls -lt traces/ | head
python -c "
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    print(r['type'], r.get('step_id'), r.get('tool_name', ''))
" traces/episode_<hex>.jsonl
```

Tests:

```bash
python -m pytest tests/
```

## 17. Guidance For Future Sessions

If picking this up later:

- The real runtime path is `run_episode` recursing into itself for subagents. Don't introduce a second orchestrator.
- `engine.py` is the control layer; keep the step loop prose-readable.
- `llm.py` is the first place to look when model behavior looks wrong.
- Trace JSONL is the source of truth for debugging behavior.
- Config should change when execution strategy / latency / trace shape changes. Not for convenience.

Invariants to preserve:

- One step = one assistant turn.
- One `.jsonl` per episode.
- Tool calls are explicit events.
- Neutral history format (user / assistant-with-tool_calls / tool).
- `spawn_subagent` is just a tool; subagents are recursive `run_episode` calls.
- No runtime globals; everything flows through `cfg`, `episode`, `context`.
