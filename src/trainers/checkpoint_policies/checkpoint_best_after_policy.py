from typing import Any
from trainers.checkpoint_policies.checkpoint_policy import CheckpointPolicy


class CheckpointBestAfterPolicy(CheckpointPolicy):
    def __init__(self, criterion: str, after: int, max_acceptable: float = float("inf")) -> None:
        super().__init__()

        self._criterion_key: str = criterion
        self._start_batch: int = after

        self._best: float = max_acceptable

    def get_state(self) -> Any:
        return {
            "best": self._best,
        }

    def set_state(self, state: Any):
        self._best = state["best"]

    def check(self, batch: int, details: dict[str, float]) -> bool:
        if batch < self._start_batch:
            return False

        if self._criterion_key not in details:
            return False
        
        if self._best < details[self._criterion_key]:
            return False
        self._best = details[self._criterion_key]
        return True
