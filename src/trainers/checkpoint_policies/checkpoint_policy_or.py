from typing import Any
from trainers.checkpoint_policies.checkpoint_policy import CheckpointPolicy


class CheckpointPolicyOr(CheckpointPolicy):
    def __init__(self, policy1: CheckpointPolicy, policy2: CheckpointPolicy) -> None:
        super().__init__()

        self._policy1 = policy1
        self._policy2 = policy2

    def get_state(self) -> Any:
        return {
            "policy1": self._policy1.get_state(),
            "policy2": self._policy2.get_state()
        }

    def set_state(self, state: Any):
        self._policy1.set_state(state["policy1"])
        self._policy2.set_state(state["policy2"])

    def check(self, batch: int, details: dict[str, float]) -> bool:
        result1: bool = self._policy1.check(batch, details)
        result2: bool = self._policy2.check(batch, details)
        return result1 or result2
