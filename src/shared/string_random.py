import random as _random

class StringRandom:
    def __init__(self, 
                 generator: _random.Random,
                 length: int,
                 allow_duplicate: bool = False,
                 characters: str = "abcdefghijklmnopqrstuvwxyz") -> None:
        self._random: _random.Random = generator

        self._history: set[str] | None = set()
        if allow_duplicate:
            self._history = None
        
        self._characters = characters
        self._length = length
    
    def _next_character_list(self):
        for _ in range(self._length):
            yield self._random.choice(self._characters)

    def next(self) -> str:
        result_list = self._next_character_list()
        result = "".join(result_list)

        if self._history is None:
            return result
        
        if result in self._history:
            return self.next()
        
        self._history.add(result)
        return result
