from typing import Any, Dict, List, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.utils import pre_init
from pydantic import Field


class RITS(LLM):
    model_name: str = Field("meta-llama/llama-3-1-70b-instruct", alias="model")
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

        values["base_url"] = f'{values["base_url"]}/{values["model"].split("/")[1]}'
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
        return "rits"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input (.invoke)."""
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
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                url=f"{self.rits_base_url}/v1/completions",
                headers={"RITS_API_KEY": self.rits_api_key},
                json={"prompt": prompt, "stop": stop, "model": self.model_name, **params},
            )

        if response.status_code != 200:
            raise ValueError(f"Failed to call RITS: {response.text} with status code {response.status_code}")

        response_json = response.json()
        return response_json["choices"][0]["text"]
