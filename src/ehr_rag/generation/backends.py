"""Generation backends: ``anthropic``, ``openai_compatible`` and ``hf_local``.

Every backend exposes the same small interface (see ``Backend``). The prompt is
always sent as a single user message. API backends retry transient errors with
exponential backoff (SDK-internal retries are disabled so all retries are ours);
``hf_local`` does not retry (local failures are not transient).
"""

from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")

MAX_ATTEMPTS = 5
BASE_DELAY_S = 2.0
MAX_DELAY_S = 60.0
CHARS_PER_TOKEN = 4  # rough estimate, used only where no tokenizer is available


class BackendError(RuntimeError):
    """A backend cannot be built or a call failed permanently."""


@dataclass
class Completion:
    """One model response."""

    text: str
    model: str                      # model id as reported by the API / loaded locally
    revision: str | None            # resolved HF commit hash (hf_local only)
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    temperature_applied: bool       # False if the backend could not send a temperature
    templated_prompt: str | None = None  # exact string fed to the model (hf_local)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict."""
        return asdict(self)


class Backend(Protocol):
    """Interface shared by all generation backends."""

    name: str
    model: str

    def revision(self) -> str | None:
        """Resolved model revision (commit hash) if the backend knows it, else None."""

    def count_tokens(self, prompt: str) -> tuple[int, bool]:
        """(input tokens of ``prompt`` as the model sees it, exact?)."""

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Completion:
        """Generate a completion for one user message."""


def is_transient(exc: BaseException) -> bool:
    """Rate limits, timeouts, connection errors and 5xx are transient; everything else is not."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or status >= 500
    return type(exc).__name__ in ("APIConnectionError", "APITimeoutError")


def with_retries(
    fn: Callable[[], T],
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> T:
    """Call ``fn``; retry transient errors with exponential backoff; re-raise otherwise.

    After ``max_attempts`` transient failures a ``BackendError`` is raised.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - classified below
            if not is_transient(e):
                raise
            if attempt == max_attempts:
                raise BackendError(f"persistent transient failure after {max_attempts} attempts: {e!r}") from e
            delay = min(base_delay * 2 ** (attempt - 1) + random.uniform(0, 1), MAX_DELAY_S)
            log(f"transient error ({type(e).__name__}); retry {attempt}/{max_attempts - 1} in {delay:.1f}s")
            sleep(delay)
    raise AssertionError("unreachable")


def _api_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise BackendError(f"environment variable {env_name} (generation.api_key_env) is not set or empty")
    return key


def _approx_tokens(prompt: str) -> tuple[int, bool]:
    return (len(prompt) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN, False


class AnthropicBackend:
    """Anthropic Messages API.

    The pinned SDK (and current Claude models) do not accept a ``temperature``
    parameter, so none is sent and ``temperature_applied`` is recorded as False.
    """

    name = "anthropic"

    def __init__(self, model: str, api_key_env: str) -> None:
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic(api_key=_api_key(api_key_env), max_retries=0)

    def revision(self) -> str | None:
        return None

    def count_tokens(self, prompt: str) -> tuple[int, bool]:
        return _approx_tokens(prompt)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Completion:
        def call() -> Any:
            return self._client.messages.create(
                model=self.model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}]
            )

        resp = with_retries(call)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(
            text=text, model=resp.model, revision=None, input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens, finish_reason=resp.stop_reason, temperature_applied=False,
        )


class OpenAICompatibleBackend:
    """OpenAI SDK chat completions with a configurable ``base_url``."""

    name = "openai_compatible"

    def __init__(self, model: str, api_key_env: str, base_url: str | None) -> None:
        import openai

        self.model = model
        self._client = openai.OpenAI(api_key=_api_key(api_key_env), base_url=base_url, max_retries=0)

    def revision(self) -> str | None:
        return None

    def count_tokens(self, prompt: str) -> tuple[int, bool]:
        return _approx_tokens(prompt)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Completion:
        def call() -> Any:
            return self._client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=max_tokens,
            )

        resp = with_retries(call)
        choice = resp.choices[0]
        usage = resp.usage
        return Completion(
            text=choice.message.content or "", model=resp.model, revision=None,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason, temperature_applied=True,
        )


def greedy_generation_config(base: Any, max_new_tokens: int) -> Any:
    """Copy of the model's ``GenerationConfig`` with only the sampling settings overridden.

    ``do_sample=False``; temperature / top_p / top_k removed; ``max_new_tokens`` set.
    The model's own ``eos_token_id`` (a list for Llama 3.1: several end-of-turn
    tokens) is kept; ``pad_token_id`` is set to the first eos id.
    """
    cfg = copy.deepcopy(base)
    cfg.do_sample = False
    cfg.temperature = None
    cfg.top_p = None
    cfg.top_k = None
    cfg.max_new_tokens = max_new_tokens
    eos = cfg.eos_token_id
    if eos is None or (isinstance(eos, list) and not eos):
        raise BackendError("model generation_config has no eos_token_id")
    cfg.pad_token_id = eos[0] if isinstance(eos, list) else eos
    return cfg


class HFLocalBackend:
    """In-process Hugging Face transformers model: bf16 on CUDA, chat template, greedy decoding.

    The tokenizer (and config, for the revision) load lazily on first use and do
    not need a GPU; the model loads once per process on the first ``generate``
    call and requires CUDA (no CPU fallback). With ``HF_HUB_OFFLINE=1`` everything
    is read from the local HF cache.
    """

    name = "hf_local"

    def __init__(self, model: str) -> None:
        self.model = model
        self._tok: Any = None
        self._model: Any = None
        self._revision: str | None = None

    def _tokenizer(self) -> Any:
        if self._tok is None:
            from transformers import AutoConfig, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model)
            h = getattr(AutoConfig.from_pretrained(self.model), "_commit_hash", None)
            if not h:
                raise BackendError(f"could not resolve the Hugging Face revision hash for {self.model}")
            self._revision = str(h)
        return self._tok

    def _load_model(self) -> Any:
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            if not torch.cuda.is_available():
                raise BackendError("hf_local requires CUDA, but torch.cuda.is_available() is False")
            self._tokenizer()
            self._model = AutoModelForCausalLM.from_pretrained(self.model, dtype=torch.bfloat16, device_map="cuda")
            self._model.eval()
            loaded = getattr(self._model.config, "_commit_hash", None)
            if loaded and loaded != self._revision:
                raise BackendError(f"tokenizer revision {self._revision} != model revision {loaded}")
        return self._model

    def revision(self) -> str | None:
        self._tokenizer()
        return self._revision

    def template(self, prompt: str) -> str:
        """The chat-templated string (single user turn + generation prompt)."""
        return self._tokenizer().apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def _encode(self, templated: str) -> list[int]:
        # The chat template already contains the BOS token.
        return self._tokenizer()(templated, add_special_tokens=False)["input_ids"]

    def count_tokens(self, prompt: str) -> tuple[int, bool]:
        return len(self._encode(self.template(prompt))), True

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Completion:
        import torch

        if temperature != 0:
            raise BackendError("hf_local decodes greedily; generation.temperature must be 0")
        model = self._load_model()
        tok = self._tokenizer()
        templated = self.template(prompt)
        ids = torch.tensor([self._encode(templated)], device=model.device)
        gen_cfg = greedy_generation_config(model.generation_config, max_tokens)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=gen_cfg)
        new = out[0, ids.shape[1]:].tolist()
        eos = gen_cfg.eos_token_id if isinstance(gen_cfg.eos_token_id, list) else [gen_cfg.eos_token_id]
        finish = "eos" if new and new[-1] in eos else "length"
        return Completion(
            text=tok.decode(new, skip_special_tokens=True), model=self.model, revision=self._revision,
            input_tokens=int(ids.shape[1]), output_tokens=len(new), finish_reason=finish,
            temperature_applied=True, templated_prompt=templated,
        )


def make_backend(gen: dict[str, Any]) -> Backend:
    """Build the backend named by ``generation.backend`` (config must already be validated)."""
    b = gen["backend"]
    if b == "anthropic":
        return AnthropicBackend(gen["model"], gen["api_key_env"])
    if b == "openai_compatible":
        return OpenAICompatibleBackend(gen["model"], gen["api_key_env"], gen["base_url"])
    if b == "hf_local":
        return HFLocalBackend(gen["model"])
    raise BackendError(f"unknown backend {b!r}")


def gpu_name() -> str | None:
    """``torch.cuda.get_device_name(0)`` if CUDA is available, else None."""
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
