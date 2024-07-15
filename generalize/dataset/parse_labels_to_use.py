from typing import List, Optional, Tuple


def parse_labels_to_use(
    labels_to_use: Optional[List[str]], unique_labels: List[str]
) -> Tuple[list, dict, str]:
    """
    Parse list of labels (optionally including collapsing instructions)
    and return an updated list of labels to use, a collapse map
    and (when 2 classes) the positive label.

    TODO: Improve these docs. Give examples and explain why this is cool.

    Parameters
    ----------
    labels_to_use
        List of strings with labels to use, or `None`.
        One of the labels can be "-1", which will create
            a label group called "Rest" with the non-specified
            labels.
        Combine multiple labels to a single label group (e.g. cancer <- colon,rectal,prostate)
            by giving a name and the paranthesis-wrapped, comma-separated labels.
            E.g. 'cancer(colon,rectal,prostate)'.
        When `None`, the unique labels are all used, and the second
        label in `unique_labels` is considered the positive label
        (when it contains 2 labels in total).
    unique_labels
        List of all the unique labels.

    Returns
    -------
    list
        Updated list of all labels to use.
    dict
        A collapse_map mapping `name->labels`.
    str
        The positive label (class index) after collapsings etc.
            Will be the second element in `labels_to_use`.
            When >2 classes, this will be `None`.
    """
    # Select the samples to use in the analysis
    # This can include collapsing multiple labels to a single label

    if labels_to_use is None:
        return (
            unique_labels,
            None,
            None if len(unique_labels) != 2 else unique_labels[1],
        )

    # Remove empty strings (not sure this can happen but sanity check)
    labels_to_use = [label for label in labels_to_use if label]
    assert (
        len(labels_to_use) >= 2
    ), "When `labels_to_use` is specified, 2 or more labels must be specified, separated by whitespace."

    collapse_map = {}
    all_labels_to_use = []

    # Extract positive label
    positive_label = None
    if len(labels_to_use) == 2:
        positive_label = labels_to_use[1]
        if positive_label == "-1":
            positive_label = "Rest"
        elif "(" in positive_label:
            positive_label = positive_label.split("(")[0]

    # Find collapsings and post-collapse labels
    assign_rest_to_group = False
    for group in labels_to_use:
        if group == "-1":
            if assign_rest_to_group:
                raise ValueError("`labels_to_use` contained more than one `-1`.")
            assign_rest_to_group = True
            continue
        if "(" in group:
            group_name, group = group.split("(")
            # Remove end paranthesis
            assert group[-1] == ")", "When label contains `(` it must end with `)`"
            group = group[:-1]
            labels_in_group = group.split(",")
            # Trim leading and trailing whitespaces
            labels_in_group = [label.strip() for label in labels_in_group]
            assert (
                len(labels_in_group) > 0
            ), "Found no comma-separated labels within the parantheses."
            collapse_map[group_name] = labels_in_group
            all_labels_to_use += labels_in_group
        else:
            all_labels_to_use.append(group)

    # When user specified -1, we add the remaining labels as a single group
    if assign_rest_to_group:
        rest_group_labels = list(set(unique_labels).difference(set(all_labels_to_use)))
        all_labels_to_use += rest_group_labels
        if len(rest_group_labels) > 1:
            rest_group_name = "Rest"
            while rest_group_name in collapse_map.keys():
                rest_group_name += "_"
            collapse_map[rest_group_name] = rest_group_labels

    assert len(set(all_labels_to_use)) == len(
        all_labels_to_use
    ), "A label in `labels_to_use` was found more than once."

    return all_labels_to_use, collapse_map, positive_label
