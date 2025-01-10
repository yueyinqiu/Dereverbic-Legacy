from .imports import *

_path: Path | None = None
_values: dict = {}

global_variables: dict = {}

def enable(path: Path | str):
    global _path
    if _path is not None:
        raise RuntimeError(f"Dirty inspector cannot be enabled twice.")
    print("# Dirty inspector is enabled. Note that it is only for debugging purpose.")
    _path = Path(path)
    csdir.create_directory(_path.parent)

_T = TypeVar("_T")
def set(key: str, value: _T, converter: Callable[[_T], Any]):
    if _path is None:
        return
    if key in _values:
        raise KeyError(f"The key ({key}) already exists. Please try another key.")
    _values[key] = converter(value)

def set_tensor(key: str, value: Tensor):
    set(key, value, lambda x: x.clone().detach().cpu())

def set_float(key: str, value: float):
    set(key, value, lambda x: x)

def save_and_exit():
    if _path is None:
        return
    torch.save(_values, _path)
    print("# Program will exit by dirty inspector...")
    print(f"# Inspector output: {_path}")
    exit()
    