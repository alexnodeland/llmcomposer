import asyncio

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from llmcomposer.agent import CompositionDeps, composer_agent

VALID_ABC = "X:1\nT:test tune\nM:4/4\nL:1/8\nQ:1/4=80\nK:C\nCD EF G2 AB | c8 |]\n"


def test_agent_returns_validated_score_update():
    model = TestModel(custom_output_args={"reply": "here it is", "abc": VALID_ABC})
    result = asyncio.run(
        composer_agent.run("something gentle", deps=CompositionDeps(), model=model)
    )
    assert result.output.reply == "here it is"
    assert result.output.abc.startswith("X:1")


def test_invalid_abc_triggers_model_retry():
    attempts: list[bool] = []

    def flaky_composer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # First attempt returns garbage; the retry returns a valid score.
        saw_retry = any(
            isinstance(part, RetryPromptPart)
            for message in messages
            for part in message.parts
        )
        attempts.append(saw_retry)
        abc = VALID_ABC if saw_retry else "not a score at all"
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"reply": "ok", "abc": abc},
                )
            ]
        )

    result = asyncio.run(
        composer_agent.run(
            "anything", deps=CompositionDeps(), model=FunctionModel(flaky_composer)
        )
    )
    assert attempts == [False, True]
    assert result.output.abc.startswith("X:1")


def test_current_score_is_injected_into_instructions():
    captured: list[str] = []

    def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.append(messages[-1].instructions or "")
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={"reply": "ok", "abc": VALID_ABC},
                )
            ]
        )

    deps = CompositionDeps(current_abc=VALID_ABC)
    asyncio.run(composer_agent.run("brighter", deps=deps, model=FunctionModel(capture)))
    assert "<current_score>" in captured[0]
    assert "T:test tune" in captured[0]
