from conftest import load_module

memory_mod = load_module("memory", "memory.py")
llm = load_module("llm", "llm.py")


def test_detect_provider_uses_prefix_only():
    # Known OpenAI prefixes auto-detect.
    assert llm.detect_provider("gpt-4o-mini") == "openai"
    assert llm.detect_provider("o1-preview") == "openai"
    # The slash carries no routing meaning. HF-style ids fall through to the
    # default; users set backend="custom" explicitly for non-OpenAI providers.
    assert llm.detect_provider("Qwen/Qwen2.5-7B-Instruct") == "openai"
    assert llm.detect_provider("meta-llama/Llama-3.1-8B-Instruct") == "openai"


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
    test_detect_provider_uses_prefix_only()
    test_messages_to_openai_keeps_text_roles()
    test_openai_message_builder_uses_task_input()
    print("All llm tests passed.")


if __name__ == "__main__":
    main()
