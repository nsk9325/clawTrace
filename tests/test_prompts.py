from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(module_name: str, file_name: str):
    module_path = Path(__file__).resolve().parent.parent / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


config = _load_module("config", "config.py")
prompts = _load_module("prompts", "prompts.py")


def _cfg(**overrides):
    data = {**config.DEFAULT_CONFIG, **overrides}
    return config.RunConfig.from_dict(data)


def test_minimal_returns_static_string_regardless_of_flags():
    cfg = _cfg(
        system_prompt="minimal",
        allow_parallel_tools=True,
        enable_subagents=True,
    )
    out = prompts.build_system_prompt(cfg)
    assert "ClawTrace" in out
    assert "Parallel tool use" not in out
    assert "Subagents" not in out


def test_agent_substitutes_environment():
    cfg = _cfg(system_prompt="agent")
    out = prompts.build_system_prompt(cfg)
    assert "# Environment" in out
    assert "Working directory:" in out
    assert "Platform:" in out
    assert "Date:" in out


def test_agent_includes_parallel_block_only_when_enabled():
    on = prompts.build_system_prompt(_cfg(system_prompt="agent", allow_parallel_tools=True))
    off = prompts.build_system_prompt(_cfg(system_prompt="agent", allow_parallel_tools=False))
    assert "Parallel tool use" in on
    assert "Parallel tool use" not in off


def test_agent_includes_subagents_block_only_when_enabled():
    on = prompts.build_system_prompt(_cfg(system_prompt="agent", enable_subagents=True))
    off = prompts.build_system_prompt(_cfg(system_prompt="agent", enable_subagents=False))
    assert "spawn_subagent" in on
    assert "spawn_subagent" not in off


def test_subagent_path_returns_minimal_regardless_of_main_choice():
    cfg = _cfg(
        system_prompt="agent",
        allow_parallel_tools=True,
        enable_subagents=True,
    )
    out = prompts.build_system_prompt(cfg, is_subagent=True)
    assert "# Environment" not in out
    assert "Parallel tool use" not in out


def test_append_system_prompt_concatenated_after_template():
    cfg = _cfg(system_prompt="minimal", append_system_prompt="Extra rule.")
    out = prompts.build_system_prompt(cfg)
    assert out.rstrip().endswith("Extra rule.")


def test_raw_string_override_is_used_verbatim():
    raw = "Custom prompt with {literal} braces and no formatting."
    cfg = _cfg(system_prompt=raw)
    out = prompts.build_system_prompt(cfg)
    assert out.rstrip() == raw.rstrip()

    cfg2 = _cfg(system_prompt=raw, append_system_prompt="Tail.")
    out2 = prompts.build_system_prompt(cfg2)
    assert raw.rstrip() in out2
    assert out2.rstrip().endswith("Tail.")


def main() -> None:
    test_minimal_returns_static_string_regardless_of_flags()
    test_agent_substitutes_environment()
    test_agent_includes_parallel_block_only_when_enabled()
    test_agent_includes_subagents_block_only_when_enabled()
    test_subagent_path_returns_minimal_regardless_of_main_choice()
    test_append_system_prompt_concatenated_after_template()
    test_raw_string_override_is_used_verbatim()
    print("All prompts tests passed.")


if __name__ == "__main__":
    main()
