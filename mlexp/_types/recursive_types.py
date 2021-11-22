from typing import Mapping, TypeVar, Union

T = TypeVar("T")
nested_dict_t = Union[Mapping[str, "nested_dict_t"], T]  # type: ignore  # recursive type
