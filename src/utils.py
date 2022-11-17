import itertools

import yaml
from typing import List, Optional, Any, TypeVar, Callable, Type, cast


def merge_yaml_data(filepaths):
    data = {}
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            content = yaml.load(f, yaml.Loader)
        data |= content
    return data


T = TypeVar("T")


def flatten(l: List[List[T]]) -> List[T]:
    return list(itertools.chain(*l))


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()
