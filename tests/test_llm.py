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


memory_mod = _load_module("memory", "memory.py")
llm = _load_module("llm", "llm.py")


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
    memory = memory_mod.Memory.with_initial_user(
        "Read the README file",
        system_prompt="Test system prompt",
    )
    messages = llm._build_openai_messages(memory)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Test system prompt"
    assert messages[1]["role"] == "user"
    assert "Read the README file" in messages[1]["content"]


def main() -> None:
    test_detect_provider_uses_prefix_and_explicit_provider()
    test_messages_to_openai_keeps_text_roles()
    test_openai_message_builder_uses_task_input()
    print("All llm tests passed.")


if __name__ == "__main__":
    main()
