from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import pre_init
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a RITS message dict."""
    message_dict = {"role": "", "content": message.content}

    if isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
    else:
        raise ValueError(f"Unknown message type: {type(message)}")

    if message.name:
        message_dict["name"] = message.name
    return message_dict


def _convert_dict_to_message(response_dict: Dict[str, Any]) -> BaseMessage:
    """Convert a RITS message dict to a LangChain message."""
    role = response_dict["role"]
    content = response_dict.get("content", "")

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        additional_kwargs = {}
        if tool_calls := response_dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=content)
    elif role == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=response_dict["tool_call_id"],
            name=response_dict.get("name"),
        )
    else:
        return ChatMessage(content=content, role=role)


class ChatRITS(BaseChatModel):
    model_name: str = Field("meta-llama/Llama-3.2-90B-Vision-Instruct", alias="model")
    """Model name to use."""
    rits_api_key: str = Field(default=None, alias="api_key")
    """API key for RITS."""
    rits_base_url: str = Field(
        default="https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com",
        alias="base_url",
    )
    """Base URL for RITS."""
    best_of: int = 1
    """Generates best_of completions server-side and returns the "best"."""
    frequency_penalty: float = 0
    """Penalizes repeated tokens according to frequency."""
    logit_bias: Optional[Dict[str, float]] = Field(default_factory=dict)
    """Adjust the probability of specific tokens being generated."""
    max_tokens: int = 16
    """The maximum number of tokens to generate in the completion."""
    min_tokens: int = 0
    """The minimum number of tokens to generate in the completion."""
    n: int = 1
    """How many completions to generate for each prompt."""
    presence_penalty: float = 0
    """Penalizes repeated tokens."""
    seed: Optional[int] = None
    """Seed for randomness."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    top_k: int = -1
    """Number of top tokens to consider at each step."""
    streaming: bool = False
    """Whether to stream the results or not."""
    repetition_penalty: float = 1
    """Penalizes repeated tokens."""
    length_penalty: float = 1
    """Penalizes longer completions."""
    ignore_eos: bool = False
    """Whether to ignore the eos token."""
    stop: Optional[List[str]] = None
    """Stop words to use when generating. Model output is cut off at the first occurrence of the stop substrings."""

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the environment."""
        if values["api_key"] is None:
            raise ValueError("RITS API key is required.")

        # values["base_url"] =
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Return default params for the model."""
        return {
            "best_of": self.best_of,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "n": self.n,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "ignore_eos": self.ignore_eos,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "rits-chat"

    @staticmethod
    def _convert_messages_to_dicts(messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to RITS messages."""
        return [_convert_message_to_dict(message) for message in messages]

    def _create_chat_result(self, response: Dict) -> ChatResult:
        generations = []
        for choice in response["choices"]:
            message = _convert_dict_to_message(choice["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=choice.get("finish_reason")),
            )
            generations.append(gen)

        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
            "system_fingerprint": response.get("system_fingerprint", ""),
        }

        return ChatResult(generations=generations, llm_output=llm_output)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],  # noqa: UP006
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate text."""
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop
        elif stop is None:
            stop = []

        params = {
            **self._default_params,
            **kwargs,
        }

        messages = self._convert_messages_to_dicts(messages)

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                url=f"{self.rits_base_url}/v1/chat/completions",
                headers={"RITS_API_KEY": self.rits_api_key},
                json={
                    "messages": messages,
                    "stop": stop,
                    "model": self.model_name,
                    **params,
                },
            )

        if response.status_code != 200:
            raise ValueError(f"Failed to call RITS: {response.text} with status code {response.status_code}")

        response_json = response.json()
        return self._create_chat_result(response_json)
