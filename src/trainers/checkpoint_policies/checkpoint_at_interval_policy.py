from typing import Any
from trainers.checkpoint_policies.checkpoint_policy import CheckpointPolicy


class CheckpointAtIntervalPolicy(CheckpointPolicy):
    def __init__(self, interval: int) -> None:
        super().__init__()

        self._interval = interval

    def get_state(self) -> Any:
        return None

    def set_state(self, state: Any):
        return

    def check(self, batch: int, details: dict[str, float]) -> bool:
        return batch % self._interval == 0
