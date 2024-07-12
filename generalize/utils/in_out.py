import os
import json
from typing import Tuple


def append_dict_to_json(record, path):
    """
    Append a dict object to a json file
    without loading the existing file.

    The dict is appended as a single line.
    """
    with open(path, "a") as f:
        json.dump(record, f)
        f.write(os.linesep)


def load_appended_json_dicts(path):
    """
    Load json file where dicts were appended
    with `append_record_to_json()`.
    """
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def attributes_are_jsonable(d: dict) -> bool:
    """
    Check that a key-value pair have the allowed types to be saved in a json file.
    When `val` is a dict, the types are checked recursively.
    """

    _allowed_types = (int, float, str, dict, list, bool)

    disallowed_members = find_disallowed_dict_element_types(
        d=d, val_allowed_types=_allowed_types, key_allowed_types=(str), allow_none=True
    )

    return not len(disallowed_members)


def find_disallowed_dict_element_types(
    d: dict, val_allowed_types: Tuple, key_allowed_types: Tuple, allow_none: bool = True
):
    """
    Find dict elements with disallowed types, recursively.
    """
    disallowed_elements = []
    for key, val in d.items():

        # Check type of value
        if isinstance(val, dict):
            disallowed_elements += find_disallowed_dict_element_types(
                d=val,
                val_allowed_types=val_allowed_types,
                key_allowed_types=key_allowed_types,
                allow_none=allow_none,
            )
        else:
            if val is None:
                if not allow_none:
                    disallowed_elements += [(key, None), "value"]
            elif not isinstance(val, val_allowed_types):
                disallowed_elements += [(key, type(val)), "value"]

        # Check type of key
        if not isinstance(key, key_allowed_types):
            disallowed_elements += [(key, type(key)), "key"]

    return disallowed_elements
