import pytest

from mlexp._types import nested_dict_t
from mlexp._types.converter import nested_dict2dict


@pytest.mark.parametrize(
    (
        "nested_dict",
        "dic",
    ),
    [
        # case1
        (
            3,
            {"": 3},
        ),
        # case2
        (
            {"a": 6, "b": 7},
            {"a": 6, "b": 7},
        ),
        # case3
        (
            {"a": {"1": 3, "2": 10}, "b": 7},
            {"a-1": 3, "a-2": 10, "b": 7},
        ),
    ],
)
def test_nested_dict2dict(nested_dict: nested_dict_t[int], dic: "dict"):
    produced: "dict[str, int]" = nested_dict2dict(nested_dict, key_connector="-")
    assert produced == dic
