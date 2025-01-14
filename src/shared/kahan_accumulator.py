class KahanAccumulator:
    def __init__(self, initial_value: float = 0):
        self._value: float = initial_value
        self._compensation: float = 0
    
    def value(self):
        return self._value

    def compensation(self):
        return self._compensation

    def add(self, value: float) -> 'KahanAccumulator':
        y: float = value - self._compensation
        next_value: float = self._value + y
        self._compensation = (next_value - self._value) - y
        self._value = next_value
        return self


def _test():
    accumulator: KahanAccumulator = KahanAccumulator(1)
    accumulator.add(123.12).add(219.22).add(21912).add(21.111).add(298.22)
    print(accumulator.value())
    print(accumulator.compensation())
    print(1 + 123.12 + 219.22 + 21912 + 21.111 + 298.22)


if __name__ == "__main__":
    _test()
