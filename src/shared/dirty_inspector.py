from imports import *
from .static_class import StaticClass


class DirtyInspector(StaticClass):
    _path: Path | None = None
    _values: dict = {}
    _variables: dict = {}

    @classmethod
    def variables(cls):
        return cls._variables

    @classmethod
    def enable(cls, path: Path | str):
        if cls._path is not None:
            raise RuntimeError(f"Dirty inspector cannot be enabled twice.")
        print("# Dirty inspector is enabled. Note that it is only for debugging purpose.")
        cls._path = Path(path)
        csdir.create_directory(cls._path.parent)

    _T = TypeVar("_T")  # pylint: disable=un-declared-variable
    @classmethod
    def set(cls, key: str, value: _T, converter: Callable[[_T], Any]):
        if cls._path is None:
            return
        if key in cls._values:
            raise KeyError(f"The key ({key}) already exists. Please try another key.")
        cls._values[key] = converter(value)

    @classmethod
    def set_tensor(cls, key: str, value: Tensor):
        cls.set(key, value, lambda x: x.clone().detach().cpu())

    @classmethod
    def set_float(cls, key: str, value: float):
        cls.set(key, value, lambda x: x)

    @classmethod
    def save_and_exit(cls):
        if cls._path is None:
            return
        torch.save(cls._values, cls._path)
        print("# Program will exit by dirty inspector...")
        print(f"# Inspector output: {cls._path}")
        exit()
