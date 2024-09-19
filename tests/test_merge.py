import pytest
from chataigne.messages import merge


def test_merge_lists():
    a = [1, 2, 3]
    b = [4, 5, 6]
    result = merge(a, b)
    assert result == [1, 2, 3, 4, 5, 6]


def test_merge_dicts_non_conflicting():
    a = {"key1": "value1"}
    b = {"key2": "value2"}
    result = merge(a, b)
    assert result == {"key1": "value1", "key2": "value2"}


def test_merge_dicts_conflicting_same_values():
    a = {"key1": "value1"}
    b = {"key1": "value1"}
    result = merge(a, b)
    assert result == {"key1": "value1"}


def test_merge_dicts_conflicting_different_values():
    a = {"key1": "value1"}
    b = {"key1": "value2"}
    with pytest.raises(AssertionError):
        merge(a, b)


def test_merge_nested_dicts():
    a = {"key1": {"subkey1": "subvalue1"}}
    b = {"key1": {"subkey2": "subvalue2"}}
    result = merge(a, b)
    assert result == {"key1": {"subkey1": "subvalue1", "subkey2": "subvalue2"}}


def test_merge_list_and_dict():
    a = [1, 2, 3]
    b = {"key1": "value1"}
    with pytest.raises(ValueError):
        merge(a, b)


def test_merge_nested_conflicting_different_value():
    a = {"key1": {"subkey1": "subvalue1"}}
    b = {"key1": {"subkey1": "subvalue2"}}
    with pytest.raises(AssertionError):
        merge(a, b)


def test_merge_all_at_once():
    a = {"key1": {"subkey1": "subvalue1"}, "key2": "value2"}
    b = {"key1": {"subkey2": "subvalue2"}, "key3": "value3"}
    result = merge(a, b)
    assert result == {
        "key1": {"subkey1": "subvalue1", "subkey2": "subvalue2"},
        "key2": "value2",
        "key3": "value3",
    }


def test_merge_empty_dict():
    a = {"key1": "value1"}
    b = {}
    result = merge(a, b)
    assert result == {"key1": "value1"}
