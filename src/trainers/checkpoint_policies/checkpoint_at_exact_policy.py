from typing import Any
from trainers.checkpoint_policies.checkpoint_policy import CheckpointPolicy


class CheckpointAtExactPolicy(CheckpointPolicy):
    def __init__(self, batches: set[int]) -> None:
        super().__init__()
        self._batches = batches

    def get_state(self) -> Any:
        return None

    def set_state(self, state: Any):
        return

    def check(self, batch: int, details: dict[str, float]) -> bool:
        return batch in self._batches
