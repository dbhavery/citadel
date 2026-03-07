"""Tests for citadel_gateway.models — request validation and response serialization."""

from citadel_gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceMessage,
    Usage,
)


class TestChatCompletionRequest:
    """Request model validation."""

    def test_minimal_request(self) -> None:
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert req.model == "gpt-4"
        assert len(req.messages) == 1
        assert req.stream is False
        assert req.temperature is None

    def test_full_request(self) -> None:
        req = ChatCompletionRequest(
            model="claude-sonnet",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hi"),
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
            stream=True,
            stop=["\n"],
        )
        assert req.temperature == 0.7
        assert req.max_tokens == 1024
        assert req.stream is True
        assert req.stop == ["\n"]


class TestChatCompletionResponse:
    """Response serialization matches OpenAI format."""

    def test_response_has_required_fields(self) -> None:
        resp = ChatCompletionResponse(
            model="gpt-4",
            choices=[
                Choice(
                    index=0,
                    message=ChoiceMessage(role="assistant", content="Hi there!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        data = resp.to_openai_dict()

        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-4"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hi there!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 5
        assert data["usage"]["completion_tokens"] == 3
        assert data["usage"]["total_tokens"] == 8

    def test_response_id_format(self) -> None:
        resp = ChatCompletionResponse(model="test")
        assert resp.id.startswith("chatcmpl-")

    def test_response_created_is_int(self) -> None:
        resp = ChatCompletionResponse(model="test")
        assert isinstance(resp.created, int)
        assert resp.created > 0
