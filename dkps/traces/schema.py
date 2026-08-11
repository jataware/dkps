from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    """One agent step: a tool call (with the assistant text that produced it, if any)
    or a text-only assistant turn."""
    index: int
    assistant_text: str | None = None
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    tool_output: str | None = None
    tool_success: bool | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass
class Trace:
    """One (model, query, replicate) trajectory. Neutral representation --
    loaders for other logging formats should produce this."""
    model_id: str
    query_id: str
    replicate: int
    steps: list[Step] = field(default_factory=list)
    final_output: str | None = None
    exit_status: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.tool_name is not None)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.model_id, self.query_id, self.replicate)
