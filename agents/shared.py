"""Shared utilities used by both form agents."""

import json
import yaml
from pathlib import Path

from pydantic import BaseModel, Field


def load_template(path: str) -> dict:
    """Load a form template from JSON or YAML."""
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return json.loads(text)


class JudgeResult(BaseModel):
    sufficient: bool = Field(description="Whether the answer meets the criteria")
    feedback: str = Field(description="Why the answer is or isn't sufficient, with guidance on how to improve")
