from __future__ import annotations

import tempfile
from pathlib import Path

from conftest import load_module

tools = load_module("clawtrace_tools", "tools.py")


def test_builtin_tools_are_registered():
    assert tools.get_tool("read_file") is not None
    assert tools.get_tool("bash") is not None


def test_read_file_tool_reads_lines(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("alpha\nbeta\n", encoding="utf-8")

    result = tools.execute_tool("read_file", {"file_path": str(file_path)})

    assert "alpha" in result
    assert "beta" in result
    # Line count is surfaced so the agent can plan pagination without
    # blindly trying offsets past EOF.
    assert "file has 2 lines" in result


def test_read_file_returns_clear_error_when_offset_past_eof(tmp_path):
    file_path = tmp_path / "short.txt"
    file_path.write_text("only\ntwo\n", encoding="utf-8")
    result = tools.execute_tool(
        "read_file",
        {"file_path": str(file_path), "offset": 100, "limit": 10},
    )
    assert result.startswith("Error:")
    assert "past EOF" in result
    assert "2 lines" in result


def test_read_file_distinguishes_truly_empty_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    result = tools.execute_tool("read_file", {"file_path": str(file_path)})
    assert result == "(file is empty)"


def test_bash_tool_runs_command():
    result = tools.execute_tool("bash", {"command": "printf hello"})
    assert result == "hello"


def test_bash_denylist_refuses_kill_and_sudo():
    for command in ["sudo rm -rf /", "kill 1234", "pkill -f vllm", "killall python", "systemctl stop nginx"]:
        result = tools.execute_tool("bash", {"command": command})
        assert result.startswith("Error:")
        assert "refused" in result.lower()


def test_bash_denylist_allows_innocent_substring_matches():
    # `skilled`, `killer-app`, comment containing the word — these should not be refused.
    result = tools.execute_tool("bash", {"command": "echo 'I am skilled at echo'"})
    assert "skilled" in result


def test_bash_env_overrides_inject_into_subprocess():
    from types import SimpleNamespace
    overrides = {"CLAWTRACE_TEST_FLAG": "from-override"}
    result = tools.execute_tool(
        "bash",
        {"command": "echo $CLAWTRACE_TEST_FLAG"},
        context=SimpleNamespace(env_overrides=overrides),
    )
    assert result.strip() == "from-override"


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
