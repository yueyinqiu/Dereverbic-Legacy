from typing import TypedDict


class KahanAccumulator:
    def __init__(self, initial_value: float = 0.):
        self._value: float = initial_value
        self._compensation: float = 0.
    
    def value(self):
        return self._value

    def compensation(self):
        return self._compensation

    def add(self, value: float) -> 'KahanAccumulator':
        compensated: float = value - self._compensation
        next_value: float = self._value + compensated
        self._compensation = next_value - self._value - compensated
        self._value = next_value
        return self
    
    class StateDict(TypedDict):
        value: float
        compensation: float

    def get_state(self) -> StateDict:
        return {
            "value": self._value,
            "compensation": self._compensation
        }

    def set_state(self, state: StateDict):
        self._value = state["value"]
        self._compensation = state["compensation"]


def _test():
    values: list[float] = [1e10] + [1e-10] * 1000000
    kahan: KahanAccumulator = KahanAccumulator()
    normal: float = 0

    value: float
    for value in values:
        kahan.add(value)
        normal += value

    print(kahan.value())
    print(kahan.compensation())
    print(normal)


if __name__ == "__main__":
    _test()
