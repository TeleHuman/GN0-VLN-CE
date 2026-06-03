#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prompt templates for BAE navigation agent."""

from __future__ import annotations

PROMPT_V3HF = r"""<image><image>
You are given TWO images from the same navigation episode.
Image 1: HISTORY mosaic (4x4), NEW->OLD sampled by discrete action steps (current excluded; missing => black).
Image 2: CURRENT first-person RGB.

TASK_TYPE: PLANNING
Instruction: "{INSTRUCTION_TEXT}"

Your task:
- Actions: exactly 6 tokens from {{<FWD>,<LEFT>,<RIGHT>,<STOP>}}.
    - Action semantics: <FWD> moves forward 0.25m; <LEFT>/ <RIGHT> rotate 15 degrees in place (no translation); <STOP> means stop.

Output (STRICT): JSON with key "vlnce".
- "vlnce": "<action><FWD>,...,<STOP></action>"
No extra text.
"""

def _render_prompt(prompt: str, **kwargs) -> str:
    return prompt.format(**kwargs)


def build_prompt(
    prompt_type: str,
    instruction: str,
    cur_x: int | None = None,
    cur_y: int | None = None,
    occ_w: int | None = None,
    occ_h: int | None = None,
    occ_meter_per_px: float = 0.05,
    occ_rot_deg: int = 0,
    prev_actions: str | None = None,
    **_: object,
) -> str:
    """Build the V3HF prompt string.

    The aligned collect/eval codepaths still call this helper with the legacy
    V1/V2/V3-era OCC arguments. Keep a signature-compatible shim here so the
    simplified V3HF-only submit branch remains runtime-compatible.
    """
    pt = str(prompt_type).upper().strip()
    if pt != "V3HF":
        raise ValueError(f"Unsupported prompt type: {prompt_type!r}. Only 'V3HF' is supported.")
    prev_actions = prev_actions or "<action><None>,<None>,<None>,<None>,<None></action>"
    return _render_prompt(
        PROMPT_V3HF,
        INSTRUCTION_TEXT=instruction,
        PREV_ACTIONS=prev_actions,
    )
