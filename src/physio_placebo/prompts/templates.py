"""Three prompt templates × {0-shot, 4-shot}. IDs are not frozen until the sweep."""

from __future__ import annotations

from collections.abc import Sequence

from physio_placebo.scoring.logprob import load_label_spec

TEMPLATE_IDS = ("instruction", "clinical", "minimal")


def _meanings(dataset: str) -> dict[str, str]:
    spec = load_label_spec()
    return spec["datasets"][dataset]


def _choice_line(dataset: str) -> str:
    m = _meanings(dataset)
    return f"A = {m['A']}. B = {m['B']}. Answer with a single token, A or B."


TEMPLATES: dict[str, str] = {
    "instruction": (
        "Classify this physiological recording.\n{choice}\n"
        "{exemplars}{body}\nAnswer:"
    ),
    "clinical": (
        "You are scoring a 60-second wearable window for a binary state.\n{choice}\n"
        "{exemplars}{body}\nAnswer:"
    ),
    "minimal": "{choice}\n{exemplars}{body}\nAnswer:",
}


def render_prompt(
    template_id: str,
    dataset: str,
    body: str,
    exemplars: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Render a prompt. ``exemplars`` is a list of (body, token) pairs for k-shot."""
    if template_id not in TEMPLATES:
        raise ValueError(f"unknown template_id {template_id!r}")
    shots = ""
    if exemplars:
        parts = []
        for ex_body, token in exemplars:
            parts.append(f"{ex_body}\nAnswer: {token}\n")
        shots = "Examples:\n" + "".join(parts) + "\n"
    return TEMPLATES[template_id].format(
        choice=_choice_line(dataset),
        exemplars=shots,
        body=body,
    )
