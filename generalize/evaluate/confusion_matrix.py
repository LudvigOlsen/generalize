from typing import Callable, Optional, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix
from abc import ABC, abstractmethod
from nattrs import nested_setattr, nested_getattr
from .confusion_matrices import ConfusionMatrices

from generalize.evaluate.prepare_inputs import BinaryPreparer


class BaseConfusionMatrix(ABC):

    def __init__(self, note: Optional[str] = None) -> None:
        super().__init__()
        self.confusion_matrix = None
        self.note = note

    def get_counts(self):
        """
        Gets the confusion matrix.
        """
        return self.confusion_matrix

    @abstractmethod
    def _operate(self, other, fn: Callable, fn_name: str):
        pass

    def __radd__(self, other):
        """
        Required to enable `sum()` on a list of objects.
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other):
        return self._operate(other=other, fn=lambda x, y: x + y, fn_name="add")

    def __sub__(self, other):
        return self._operate(other=other, fn=lambda x, y: x - y, fn_name="subtract")

    def __mul__(self, other):
        return self._operate(other=other, fn=lambda x, y: x * y, fn_name="multiply")

    def __div__(self, other):
        return self._operate(other=other, fn=lambda x, y: x / y, fn_name="divide")

    @staticmethod
    def _check_assign_attribute(obj1, obj2, new_obj, attribute) -> None:
        """
        Check that an attribute is equal in two objects and assign the value
        to a new object.

        Utility used in `._operate()` methods.
        """
        # Get attribute values
        obj1_attr = nested_getattr(obj1, attr=attribute)
        obj2_attr = nested_getattr(obj2, attr=attribute)

        # Check they are equal in the two objects
        none_check = obj1_attr is None and obj2_attr is None
        array_check = (
            isinstance(obj1_attr, np.ndarray) and (obj1_attr == obj2_attr).all()
        )
        list_dict_check = isinstance(obj1_attr, (list, dict)) and obj1_attr == obj2_attr
        if not (none_check or array_check or list_dict_check):
            raise ValueError(
                f"The `{obj1.__class__.__name__}` objects had different {attribute}."
            )

        # Set value in the new object
        nested_setattr(obj=new_obj, attr=attribute, value=obj1_attr)

    @staticmethod
    def _format_string_tuple(string_tuple: Tuple[str, str], initial_indent=0) -> str:
        def indent_str(s, indent=4):
            indent_str = "".join([" "] * indent)
            s = indent_str + s.replace("\n", "\n" + indent_str)
            return s

        return (
            indent_str(string_tuple[0], indent=initial_indent)
            + "\n"
            + indent_str(string_tuple[1], indent=initial_indent + 2)
        )


class BinaryConfusionMatrix(BaseConfusionMatrix):

    def __init__(self, note: Optional[str] = None) -> None:
        super().__init__(note=note)
        self.confusion_matrix = None
        self.count_names = None
        self.class_roles = None
        self.raw_class_roles = None
        self.labels = None

    def fit(self, targets, predictions, positive, labels=None):
        """
        Wrapper of sklearn's confusion_matrix which
        saves info about which class was the positive class.
        """

        # Reduce to 1D int32 arrays
        targets = BinaryPreparer.prepare_targets(targets=targets)
        predictions = BinaryPreparer.prepare_predictions(predictions=predictions)
        self.count_names = np.asarray([["TN", "FP"], ["FN", "TP"]])
        self.labels = labels
        self.confusion_matrix, self.class_roles = (
            BinaryConfusionMatrix._fit_confusion_matrix(
                targets=targets, predictions=predictions, positive=positive
            )
        )

        # Replace the class names with their labels
        self.raw_class_roles = self.class_roles.copy()
        if self.labels is not None:
            self.class_roles = {k: self.labels[v] for k, v in self.class_roles.items()}

        return self

    @property
    def classes(self):
        return list(self.class_roles.keys())

    def to_collection(self):
        """
        Convert to a one-element `ConfusionMatrices` collection.
        E.g. in order to save the confusion matrix to disk.
        """
        coll = ConfusionMatrices(
            classes=self.classes,
            class_roles=self.class_roles,
            count_names=self.count_names,
        )
        coll.add(path="Confusion Matrix", matrix=self.confusion_matrix)
        return coll

    def _operate(self, other, fn: Callable, fn_name: str):
        if not isinstance(other, BinaryConfusionMatrix):
            raise TypeError(
                f"{fn_name}: `other` had the wrong type: {type(other)}. "
                "Must have type `BinaryConfusionMatrix`."
            )

        # Initialize new object
        new_matrix = BinaryConfusionMatrix()

        # Check and transfer attributes
        attributes = ["count_names", "labels", "class_roles", "raw_class_roles"]
        for attribute in attributes:
            BaseConfusionMatrix._check_assign_attribute(
                obj1=self, obj2=other, new_obj=new_matrix, attribute=attribute
            )

        # Apply function
        new_matrix.confusion_matrix = fn(self.confusion_matrix, other.confusion_matrix)
        return new_matrix

    def __str__(self) -> str:
        out = "\n".join(
            [
                "Confusion Matrix:",
                BaseConfusionMatrix._format_string_tuple(
                    ("Counts:", str(self.confusion_matrix)), initial_indent=2
                ),
                BaseConfusionMatrix._format_string_tuple(
                    ("Count Names:", str(self.count_names)), initial_indent=2
                ),
                BaseConfusionMatrix._format_string_tuple(
                    ("Class Roles:", str(self.class_roles)), initial_indent=2
                ),
            ]
        )
        if self.note is not None:
            out += f"\n  Note: {self.note}"
        return out

    @property
    def positive(self):
        return self.class_roles["positive"]

    @property
    def negative(self):
        return self.class_roles["negative"]

    @staticmethod
    def _fit_confusion_matrix(targets, predictions, positive):
        # sklearn's `confusion_matrix()`` uses the second class
        # in alphanumeric order as the positive class
        # so we set the positive class to 1 and negative class to 0
        unique_classes = list(
            set(np.unique(targets)).union(set(np.unique(predictions)))
        )
        unique_classes = sorted(unique_classes)
        assert (
            positive in unique_classes
        ), "`positive` class was not found in `targets` nor `predictions`."

        # Ensure type is str or int, not numpy.int32!
        def class_type_fn(c):
            return c if isinstance(c, str) else int(c)

        class_roles = {
            "negative": [class_type_fn(cl) for cl in unique_classes if cl != positive][
                0
            ],
            "positive": class_type_fn(positive),
        }
        targets = (targets == positive).astype(np.int32)
        predictions = (predictions == positive).astype(np.int32)
        return confusion_matrix(targets, predictions), class_roles


class MulticlassConfusionMatrix(BaseConfusionMatrix):

    def __init__(self, note: Optional[str] = None) -> None:
        super().__init__(note=note)
        self.confusion_matrix = None
        self.raw_classes = None
        self.classes = None
        self.count_names = None
        self.labels = None

    def fit(self, targets, predictions, labels=None):
        """
        Wrapper of sklearn's confusion_matrix which
        saves the classes as well.
        """
        self.confusion_matrix, self.classes = (
            MulticlassConfusionMatrix._fit_confusion_matrix(
                targets=targets, predictions=predictions
            )
        )

        # Replace the class names with their labels
        self.raw_classes = self.classes
        if labels is not None:
            self.classes = [labels[cl] for cl in self.classes]

        # Create count names (0==0, 0==1, ...)
        self.count_names = np.asarray(
            [
                f"{c_i}=={c_j}"
                for c_i in sorted(self.classes)
                for c_j in sorted(self.classes)
            ]
        ).reshape((len(self.classes), len(self.classes)))

        return self

    def to_collection(self):
        """
        Convert to a one-element `ConfusionMatrices` collection.
        E.g. in order to save the confusion matrix to disk.
        """
        coll = ConfusionMatrices(classes=self.classes, count_names=self.count_names)
        coll.add(path="Confusion Matrix", matrix=self.confusion_matrix)
        return coll

    def _operate(self, other, fn: Callable, fn_name: str):
        if not isinstance(other, MulticlassConfusionMatrix):
            raise TypeError(
                f"{fn_name}: `other` had the wrong type: {type(other)}. "
                "Must have type `MulticlassConfusionMatrix`."
            )

        # Initialize new object
        new_matrix = MulticlassConfusionMatrix()

        # Check and transfer attributes
        attributes = ["labels", "classes", "raw_classes", "count_names"]
        for attribute in attributes:
            BaseConfusionMatrix._check_assign_attribute(
                obj1=self, obj2=other, new_obj=new_matrix, attribute=attribute
            )

        # Apply function
        new_matrix.confusion_matrix = fn(self.confusion_matrix, other.confusion_matrix)
        return new_matrix

    def __str__(self) -> str:
        out = "\n".join(
            [
                "Confusion Matrix:",
                BaseConfusionMatrix._format_string_tuple(
                    ("Counts:", str(self.confusion_matrix)), initial_indent=2
                ),
                BaseConfusionMatrix._format_string_tuple(
                    ("Classes:", str(self.classes)), initial_indent=2
                ),
                BaseConfusionMatrix._format_string_tuple(
                    ("Count Names:", str(self.count_names)), initial_indent=2
                ),
            ]
        )
        if self.note is not None:
            out += f"\n  Note: {self.note}"
        return out

    @staticmethod
    def _fit_confusion_matrix(targets, predictions):
        unique_classes = list(
            set(np.unique(targets)).union(set(np.unique(predictions)))
        )
        unique_classes = sorted(unique_classes)
        return confusion_matrix(targets, predictions), unique_classes
