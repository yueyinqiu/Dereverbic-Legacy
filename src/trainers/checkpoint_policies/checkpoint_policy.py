import abc
from typing import Any


class CheckpointPolicy(abc.ABC):
    @abc.abstractmethod
    def get_state(self) -> Any:
        ...

    @abc.abstractmethod
    def set_state(self, state: Any):
        ...

    @abc.abstractmethod
    def check(self, batch: int, details: dict[str, float]) -> bool:
        ...

    def __or__(self, value: Any) -> 'CheckpointPolicy':
        from trainers.checkpoint_policies.checkpoint_policy_or import CheckpointPolicyOr
        return CheckpointPolicyOr(self, value)
