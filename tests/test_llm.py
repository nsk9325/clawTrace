import importlib.util
import sys
from pathlib import Path


def _load_llm_module():
    llm_path = Path(__file__).resolve().parent.parent / "llm.py"
    spec = importlib.util.spec_from_file_location("clawtrace_llm", llm_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {llm_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


llm = _load_llm_module()


def test_detect_provider_uses_prefix_and_explicit_provider():
    assert llm.detect_provider("gpt-4o-mini") == "openai"
    assert llm.detect_provider("custom/my-model") == "custom"


def test_messages_to_openai_keeps_text_roles():
    messages = llm.messages_to_openai([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
        {
            "role": "assistant",
            "content": "assistant",
            "tool_calls": [
                {"id": "call_1", "name": "bash", "input": {"command": "pwd"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ignored"},
    ])

    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
        {
            "role": "assistant",
            "content": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": "{\"command\": \"pwd\"}",
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ignored"},
    ]


def test_openai_message_builder_uses_task_input():
    messages = llm._build_openai_messages([{"role": "user", "content": "Read the README file"}])

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Read the README file" in messages[1]["content"]


def main() -> None:
    test_detect_provider_uses_prefix_and_explicit_provider()
    test_messages_to_openai_keeps_text_roles()
    test_openai_message_builder_uses_task_input()
    print("All llm tests passed.")


if __name__ == "__main__":
    main()
