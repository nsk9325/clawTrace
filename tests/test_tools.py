from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


def _load_tools_module():
    tools_path = Path(__file__).resolve().parent.parent / "tools.py"
    spec = importlib.util.spec_from_file_location("clawtrace_tools", tools_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {tools_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tools = _load_tools_module()


def test_builtin_tools_are_registered():
    assert tools.get_tool("read_file") is not None
    assert tools.get_tool("bash") is not None


def test_read_file_tool_reads_lines(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\nbeta\n", encoding="utf-8")

    result = tools.execute_tool("read_file", {"file_path": str(file_path)})

    assert "alpha" in result
    assert "beta" in result


def test_bash_tool_runs_command():
    result = tools.execute_tool("bash", {"command": "printf hello"})
    assert result == "hello"


def test_execute_tool_classifies_raised_exception_as_error():
    def raiser(params, cfg, ctx):
        raise RuntimeError("boom")

    tools.register_tool(
        tools.ToolDef(
            name="_test_raiser",
            schema={
                "name": "_test_raiser",
                "description": "raises",
                "input_schema": {"type": "object", "properties": {}},
            },
            func=raiser,
        )
    )
    output = tools.execute_tool("_test_raiser", {}, {})
    assert output.startswith("Error:")
    assert "boom" in output


def main() -> None:
    test_builtin_tools_are_registered()
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_read_file_tool_reads_lines(Path(tmp_dir))
    test_bash_tool_runs_command()
    print("All tools tests passed.")


if __name__ == "__main__":
    main()
