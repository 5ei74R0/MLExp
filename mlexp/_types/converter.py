"""type converters"""
from typing import Mapping, TypeVar

from mlexp._types import nested_dict_t

T = TypeVar("T")


def nested_dict2dict(nested_dict: nested_dict_t[T], key_connector: str) -> dict[str, T]:
    def _rec(nested_dict: nested_dict_t, target: dict, key: str = "") -> None:
        if not isinstance(nested_dict, Mapping):
            target[key] = nested_dict
            return
        for k, sub_dict in nested_dict.items():
            next_key: str = key + (key_connector if key != "" else "") + k
            _rec(sub_dict, target, next_key)

    res: dict[str, T] = dict()
    _rec(nested_dict, res)
    return res
