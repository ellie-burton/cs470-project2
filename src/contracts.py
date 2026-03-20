from __future__ import annotations

from typing import Any, Protocol

from src.models import Frame, ValidationReport


class AlgorithmModule(Protocol):
    """
    Integration contract for algorithm modules.

    Required exports:
    - build_frames(input_data, config) -> list[Frame]
    - validate_result(input_data, final_state) -> ValidationReport
    """

    def build_frames(self, input_data: dict[str, Any], config: dict[str, Any]) -> list[Frame]:
        ...

    def validate_result(self, input_data: dict[str, Any], final_state: dict[str, Any]) -> ValidationReport:
        ...
