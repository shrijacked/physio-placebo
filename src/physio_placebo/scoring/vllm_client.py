"""vLLM logprob client. Importing this module does not import vLLM until load()."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import yaml

from physio_placebo.paths import configs_dir
from physio_placebo.scoring.logprob import load_label_spec
from physio_placebo.scoring.vllm_extract import extract_label_logprobs, single_token_id


def load_model_spec() -> dict:
    return yaml.safe_load((configs_dir() / "models.yaml").read_text())


def build_sampling_params(SamplingParams: Any, label_ids: list[int], spec: dict) -> Any:
    """Full-vocab softmax; request label ids if this vLLM build supports it.

    Never sets allowed_token_ids — that renormalizes and is not the OSF rule.
    """
    kwargs: dict[str, Any] = {
        "temperature": float(spec["temperature"]),
        "max_tokens": int(spec["max_tokens"]),
        "logprobs": int(spec["logprobs_topk"]),
    }
    try:
        return SamplingParams(**kwargs, sampled_logprobs_ids=label_ids)
    except TypeError:
        try:
            return SamplingParams(**kwargs, logprobs=-1)
        except (TypeError, ValueError):
            return SamplingParams(**kwargs)


class VLLMLogprobClient:
    def __init__(self, model_key: str, *, llm: Any | None = None, tokenizer: Any | None = None):
        self.model_key = model_key
        self.spec = load_model_spec()
        self.model_cfg = self.spec["models"][model_key]
        self.tokens: list[str] = list(load_label_spec()["tokens"])
        self._llm = llm
        self._tokenizer = tokenizer
        self._token_ids: dict[str, int] | None = None

    def load(self) -> None:
        from vllm import LLM

        if self._llm is None:
            self._llm = LLM(
                model=self.model_cfg["hf_id"],
                quantization=self.spec.get("quantization"),
                dtype=self.spec.get("dtype", "float16"),
                max_logprobs=max(int(self.spec["logprobs_topk"]), 32),
            )
        if self._tokenizer is None:
            self._tokenizer = self._llm.get_tokenizer()
        self._token_ids = {t: single_token_id(self._tokenizer, t) for t in self.tokens}

    def _wrap_chat(self, prompt: str) -> str:
        tok = self._tokenizer
        kwargs = dict(self.model_cfg.get("chat_kwargs") or {})
        if hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            except TypeError:
                kwargs.pop("enable_thinking", None)
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
        return prompt

    def score_prompts(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        from vllm import SamplingParams

        if self._llm is None or self._token_ids is None:
            self.load()
        assert self._llm is not None and self._token_ids is not None
        wrapped = [self._wrap_chat(p) for p in prompts]
        params = build_sampling_params(
            SamplingParams, list(self._token_ids.values()), self.spec
        )
        outputs = self._llm.generate(wrapped, params)
        tables: list[dict[str, float]] = []
        for out in outputs:
            step = out.outputs[0].logprobs[0]
            tables.append(extract_label_logprobs(step, self._token_ids))
        return tables
