def all_values_same(d):
    """
    Check that all values are the same for all keys of a dict.
    """
    if not d:
        return True  # An empty dictionary is considered to have all values the same
    iterator = iter(d.values())
    first_value = next(iterator)
    return all(value == first_value for value in iterator)
