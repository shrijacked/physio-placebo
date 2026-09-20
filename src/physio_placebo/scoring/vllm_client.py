"""vLLM logprob client. Importing this module does not import vLLM until load()."""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from typing import Any

import yaml
from PIL import Image

from physio_placebo.paths import configs_dir
from physio_placebo.scoring.client import ScoreRequest
from physio_placebo.scoring.logprob import load_label_spec
from physio_placebo.scoring.vllm_extract import extract_label_logprobs, single_token_id


def load_model_spec() -> dict:
    return yaml.safe_load((configs_dir() / "models.yaml").read_text())


def prepare_vllm_runtime() -> None:
    """This box has no ninja; FlashInfer sampler JIT then kills engine warmup.

    Logprob scoring does not need that sampler.
    """
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def llm_engine_kwargs(spec: dict, model_cfg: dict) -> dict[str, Any]:
    """LLM() kwargs. Omit null quantization — vLLM 0.29 has no bitsandbytes."""
    kw: dict[str, Any] = {
        "model": model_cfg["hf_id"],
        "dtype": spec.get("dtype", "float16"),
        "max_logprobs": max(int(spec["logprobs_topk"]), 32),
    }
    quant = spec.get("quantization")
    if quant:
        kw["quantization"] = quant
    if model_cfg.get("max_model_len") is not None:
        kw["max_model_len"] = int(model_cfg["max_model_len"])
    elif spec.get("max_model_len") is not None:
        kw["max_model_len"] = int(spec["max_model_len"])
    if model_cfg.get("limit_mm_per_prompt"):
        kw["limit_mm_per_prompt"] = dict(model_cfg["limit_mm_per_prompt"])
    return kw


def pngs_to_pils(images: Sequence[bytes]) -> list[Any]:
    return [Image.open(io.BytesIO(b)).convert("RGB") for b in images]


def vl_user_content(prompt: str, n_images: int) -> list[dict[str, Any]]:
    """Images first, then the text prompt. Chat-template placeholder list."""
    if n_images < 1:
        return [{"type": "text", "text": prompt}]
    content: list[dict[str, Any]] = [{"type": "image"} for _ in range(n_images)]
    content.append({"type": "text", "text": prompt})
    return content


def vl_placeholder_prefix(n_images: int) -> str:
    return "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in range(n_images))


def build_sampling_params(SamplingParams: Any, label_ids: list[int], spec: dict) -> Any:
    """Full-vocab softmax; request label ids if this vLLM build supports it.

    Never sets allowed_token_ids — that renormalizes and is not the OSF rule.
    """
    base: dict[str, Any] = {
        "temperature": float(spec["temperature"]),
        "max_tokens": int(spec["max_tokens"]),
    }
    attempts: tuple[dict[str, Any], ...] = (
        {**base, "logprobs": len(label_ids), "logprob_token_ids": label_ids},
        {**base, "logprobs": -1},
        {**base, "logprobs": int(spec["logprobs_topk"])},
        base,
    )
    for kwargs in attempts:
        try:
            return SamplingParams(**kwargs)
        except (TypeError, ValueError):
            continue
        except Exception as exc:
            if type(exc).__name__ != "VLLMValidationError":
                raise
            continue
    return SamplingParams(**base)


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
        prepare_vllm_runtime()
        from vllm import LLM

        if self._llm is None:
            self._llm = LLM(**llm_engine_kwargs(self.spec, self.model_cfg))
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

    def _wrap_vl(self, prompt: str, images: Sequence[bytes]) -> dict[str, Any]:
        pils = pngs_to_pils(images)
        tok = self._tokenizer
        kwargs = dict(self.model_cfg.get("chat_kwargs") or {})
        if tok is not None and hasattr(tok, "apply_chat_template"):
            try:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": vl_user_content(prompt, len(pils))}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            except TypeError:
                text = self._wrap_chat(vl_placeholder_prefix(len(pils)) + "\n" + prompt)
        else:
            text = self._wrap_chat(vl_placeholder_prefix(len(pils)) + "\n" + prompt)
        mm: Any = pils[0] if len(pils) == 1 else pils
        return {"prompt": text, "multi_modal_data": {"image": mm}}

    def score_prompts(self, prompts: Sequence[str]) -> list[dict[str, float]]:
        return self.score_requests([ScoreRequest(p) for p in prompts])

    def score_requests(self, requests: Sequence[ScoreRequest]) -> list[dict[str, float]]:
        from vllm import SamplingParams

        if self._llm is None or self._token_ids is None:
            self.load()
        assert self._llm is not None and self._token_ids is not None
        payloads: list[Any] = []
        for req in requests:
            if req.images:
                payloads.append(self._wrap_vl(req.prompt, req.images))
            else:
                payloads.append(self._wrap_chat(req.prompt))
        params = build_sampling_params(
            SamplingParams, list(self._token_ids.values()), self.spec
        )
        outputs = self._llm.generate(payloads, params)
        tables: list[dict[str, float]] = []
        for out in outputs:
            step = out.outputs[0].logprobs[0]
            tables.append(extract_label_logprobs(step, self._token_ids))
        return tables
