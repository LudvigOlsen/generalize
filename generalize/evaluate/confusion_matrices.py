"""
For handling a collection of confusion matrices from the 
same classification task.
"""

import json
from numbers import Number
import pathlib
import copy
import re
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from nattrs import nested_setattr, nested_hasattr, nested_getattr, nested_mutattr


class ConfusionMatrices:

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        class_roles: Optional[Dict[str, Union[str, int]]] = None,
        count_names: Optional[np.ndarray] = None,
        note: str = "",
    ) -> None:
        """
        Container for storing multiple confusion matrices
        *from the same classification task*.

        Parameters
        ----------
        classes
            Names of the target classes.
        class_roles
            Dictionary mapping `class role -> target class`.
            E.g. `{'negative': 'healthy', 'positive': 'cancer'}`.
        count_names
            Array of strings with a name for each count
            in the confusion matrix.
            E.g. `np.array([['TN', 'FP'], ['FN', 'TP']])`.
        note
            A string to print when printing the collection.
        """
        # NOTE: _make_base_int converts numpy integers to int
        # but doesn't affects strings, etc.
        self.classes = (
            [_make_base_int(c) for c in classes] if classes is not None else None
        )
        self.class_roles = (
            {key: _make_base_int(val) for key, val in class_roles.items()}
            if class_roles is not None
            else None
        )
        self.note = note
        self.count_names = count_names
        self.paths = []
        self._matrices = {}

        # Check class roles
        # (Got an error when `val` was numpy.int32 during JSON serialization)
        if self.class_roles is not None:
            for key, val in self.class_roles.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Keys in `class_roles` must be strings. Got: {type(key)}."
                    )
                if not isinstance(val, (int, str)):
                    raise TypeError(
                        f"Values in `class_roles` must be either integers or strings. Got: {type(val)}."
                    )

    def add(self, path: str, matrix: np.ndarray, replace: bool = False):
        """
        Add a confusion matrix to the collection.

        Parameters
        ----------
        path
            Path of dot-separated dict keys to assign `matrix` at.
            This allows the `ConfusionMatrices` collection to have a nested structure.
            For non-nested structures, this can be thought of simply as a dict key.
            See the `attr` argument and examples in `nattrs.nested_setattr()`.
        matrix
            Array of confusion matrix counts.
        replace
            Whether to replace existing arrays or
            raise a `ValueError` when the path already exists.

        Returns
        -------
        self
            Returns self to allow chaining of methods.
        """
        if not replace and nested_hasattr(self._matrices, path):
            raise ValueError(f"A matrix already exists at path: {path}.")
        nested_setattr(obj=self._matrices, attr=path, value=matrix, make_missing=True)
        if path not in self.paths:
            self.paths.append(path)

        return self

    def get(self, path: str) -> Union[dict, np.ndarray, list]:
        """
        Get array of confusion matrix counts.

        Parameters
        ----------
        path
            Path of dot-separated dict keys to get array from.
            For non-nested structures, this can be thought of simply as a dict key.
            See the `attr` argument and examples in `nattrs.nested_setattr()`.
        """
        return nested_getattr(obj=self._matrices, attr=path)

    def __len__(self):
        """
        Number of stored matrices.
        """
        return len(self.paths)

    def __radd__(self, other):
        """
        Required to enable `sum()` on a list of objects.
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __add__(self, other):
        """
        Add matrices with the matrices of
        another collection, given they have
        the same paths, or a numeric value.

        Parameters
        ----------
        other : `ConfusionMatrices` or number
            Another collection or a numeric value.

        Returns
        -------
        `ConfusionMatrices`
            New `ConfusionMatrices` collection
            with the results.
        """
        if isinstance(other, Number):

            def fn(x):
                return x + other

        else:

            def fn(x, y):
                return x + y

        return self._operate(other=other, fn=fn, fn_name="add")

    def __sub__(self, other):
        """
        Subtract matrices with the matrices of
        another collection, given they have
        the same paths, or a numeric value.

        Parameters
        ----------
        other : `ConfusionMatrices` or number
            Another collection or a numeric value.

        Returns
        -------
        `ConfusionMatrices`
            New `ConfusionMatrices` collection
            with the results.
        """
        if isinstance(other, Number):

            def fn(x):
                return x - other

        else:

            def fn(x, y):
                return x - y

        return self._operate(other=other, fn=fn, fn_name="subtract")

    def __mul__(self, other):
        """
        Multiply matrices with the matrices of
        another collection, given they have
        the same paths, or a numeric value.

        Parameters
        ----------
        other : `ConfusionMatrices` or number
            Another collection or a numeric value.

        Returns
        -------
        `ConfusionMatrices`
            New `ConfusionMatrices` collection
            with the results.
        """
        if isinstance(other, Number):

            def fn(x):
                return x * other

        else:

            def fn(x, y):
                return x * y

        return self._operate(other=other, fn=fn, fn_name="multiply")

    def __truediv__(self, other):
        """
        Divide matrices with the matrices of
        another collection, given they have
        the same paths, or a numeric value.

        Parameters
        ----------
        other : `ConfusionMatrices` or number
            Another collection or a numeric value.

        Returns
        -------
        `ConfusionMatrices`
            New `ConfusionMatrices` collection
            with the results.
        """
        if isinstance(other, Number):

            def fn(x):
                return x / other

        else:

            def fn(x, y):
                return x / y

        return self._operate(other=other, fn=fn, fn_name="divide")

    def __floordiv__(self, other):
        """
        Integer-divide matrices with the matrices of
        another collection, given they have
        the same paths, or a numeric value.

        Parameters
        ----------
        other : `ConfusionMatrices` or number
            Another collection or a numeric value.

        Returns
        -------
        `ConfusionMatrices`
            New `ConfusionMatrices` collection
            with the results.
        """
        if isinstance(other, Number):

            def fn(x):
                return x // other

        else:

            def fn(x, y):
                return x // y

        return self._operate(other=other, fn=fn, fn_name="floor divide")

    def _operate(self, other, fn: Callable, fn_name: str):
        if isinstance(other, Number):
            new_collection = self.__copy__()
            ConfusionMatrices._mutate_matrices(
                matrices=new_collection._matrices, paths=new_collection.paths, fn=fn
            )
            return new_collection
        elif not isinstance(other, ConfusionMatrices):
            raise TypeError(
                f"Can only {fn_name} the `ConfusionMatrices` with another "
                f"`ConfusionMatrices` object or a numeric value. "
                f"Got {type(ConfusionMatrices)}."
            )

        if not ConfusionMatrices.equal_settings(self, other, check_paths=True):
            raise ValueError(
                f"Cannot {fn_name} the `ConfusionMatrices` together due "
                "to different settings. Only the actual matrices can differ."
            )
        new_collection = ConfusionMatrices(
            classes=self.classes,
            class_roles=self.class_roles,
            count_names=self.count_names,
        )

        if not isinstance(self.get(path=self.paths[0]), np.ndarray) or not isinstance(
            other.get(path=self.paths[0]), np.ndarray
        ):
            raise ValueError(
                f"Cannot {fn_name} the `ConfusionMatrices` together when "
                "the matrices are not `numpy.ndarray`s. See `.to_ndarrays()`."
            )

        for path in self.paths:
            new_collection.add(
                path=path, matrix=fn(self.get(path=path), other.get(path=path))
            )

        return new_collection

    def __copy__(self):
        """
        Make a deep copy of the collection.
        """
        return copy.deepcopy(self)

    def save(self, file_path: Union[str, pathlib.PurePath]) -> None:
        """
        Write collection to disk as `.json` file.

        NOTE: The `numpy.ndarrays` are converted to lists when
        saved via json, meaning that the types are not saved.
        Upon loading, the types will be inferred by `numpy.asarray`,
        which could lead to a different dtype than the original
        arrays.

        Parameters
        ----------
        file_path
            Path to write collection's matrices to.

        Returns
        -------
        self
            Allows for chaining methods.
        """
        # TODO Fix saving+restoring of dtypes for matrices
        out = {
            "Classes": self.classes,
            "Class Roles": self.class_roles,
            "Count Names": (
                self.count_names.tolist() if self.count_names is not None else None
            ),
            "Paths": self.paths,
            "Matrices": self.to_lists(copy=True)._matrices,
            "Note": self.note,
        }
        # Write to desk
        with open(str(file_path), "w") as outfile:
            json.dump(out, outfile)

        return self

    @staticmethod
    def load(file_path: Union[str, pathlib.PurePath]):
        """
        Load a confusion matrix collection from disk and convert
        to a `ConfusionMatrices` instance.

        Parameters
        ----------
        file_path
            Path to read stored collection matrices from.
            Must be a `.json` file as written by
            `ConfusionMatrices.save()`.

        Returns
        -------
        ConfusionMatrices
            Loaded collection.
        """
        with open(str(file_path), "r") as infile:
            d = json.load(infile)
        if sorted(d.keys()) != [
            "Class Roles",
            "Classes",
            "Count Names",
            "Matrices",
            "Note",
            "Paths",
        ]:
            raise RuntimeError(
                f"Json file did not have the required top-level keys: {file_path}"
            )
        new_collection = ConfusionMatrices(
            classes=d["Classes"],
            class_roles=d["Class Roles"],
            count_names=np.asarray(d["Count Names"]),
            note=d["Note"],
        )
        new_collection.paths = d["Paths"]
        new_collection._matrices = d["Matrices"]
        new_collection.to_ndarrays(copy=False)
        return new_collection

    @staticmethod
    def merge(collections: dict, path_prefix="", note: str = ""):
        """
        Merge multiple `ConfusionMatrices` collections.

        The paths in each collection are prefixed by `<path_prefix>.<collection name>.`
        why all paths are unique during merging and no overwriting will happen.

        Parameters
        ----------
        collections : dict of `ConfusionMatrices` collections
            Mapping of collection name -> `ConfusionMatrices` object.
        path_prefix : str
            Prefix to add for all paths in the new collection.
        note : str
            A note to add to the new collection.

        Returns
        -------
        ConfusionMatrices
            A new collection containing all the collections.
            Allows chaining of methods.
        """
        if len(path_prefix) > 0 and path_prefix[-1] == ".":
            # Remove final dot
            path_prefix = path_prefix[:-1]

        # Check same attributes
        for coll_idx, (coll_name, coll) in enumerate(collections.items()):
            if coll_idx == 0:
                comp_coll = coll
                comp_coll_name = coll_name
                continue
            # Check the collections have the same settings (except paths and matrices)
            has_equal_settings, diff = ConfusionMatrices.equal_settings(
                comp_coll, coll, check_paths=False
            )
            if not has_equal_settings:
                raise ValueError(
                    f"Collections 0 ({comp_coll_name}) and "
                    f"{coll_idx} ({coll_name}) did not have "
                    f"the same settings: {diff}."
                )

        for coll_idx, (coll_name, coll) in enumerate(collections.items()):
            # Ensure column name is a string (could be integer)
            coll_name = str(coll_name)
            if "." in coll_name:
                raise ValueError(
                    "A key in `collections` contained a dot ('.'). "
                    f"This is not allowed. The key was: `{coll_name}`."
                )

            if coll_idx == 0:
                # Create new collection
                new_collection = ConfusionMatrices(
                    classes=coll.classes,
                    class_roles=coll.class_roles,
                    count_names=coll.count_names,
                    note=note,
                )

            for path in coll.paths:
                # TODO This could technically be done
                # by just wrapping in the relevant dicts
                # and assigning to _matrices and changing
                # paths - would likely be much faster
                # This could be safer though
                new_path = f"{path_prefix}.{coll_name}.{path}"
                new_collection.add(path=new_path, matrix=coll.get(path=path))

        return new_collection

    # TODO: Rename "settings" -> these are attributes
    # but not including the matrices

    @staticmethod
    def equal_settings(x1, x2, check_paths: bool = True) -> Tuple[bool, str]:
        """
        Check that the settings (not the matrices) are the
        same for two collections.

        Does not check the `note` attribute.

        Parameters
        ----------
        x1, x2
            `ConfusionMatrices` collections to test settings of.
        check_paths
            Whether to check that paths are equal.

        Returns
        -------
        bool
            Whether the two collections have equal settings.
        str
            Description of first-discovered difference between
            the two collections' settings.
        """
        if x1.classes != x2.classes:
            return False, "Different classes"
        if x1.class_roles != x2.class_roles:
            return False, "Different class roles"
        count_names_none = sum([x1.count_names is None, x2.count_names is None])
        if count_names_none == 1 or (
            count_names_none == 0 and not (x1.count_names == x2.count_names).all()
        ):
            return False, "Different count names"
        if check_paths and x1.paths != x2.paths:
            return False, "Different paths"
        return True, ""

    def to_lists(self, copy: bool = True):
        """
        Convert all matrices to lists.

        Parameters
        ----------
        copy : bool
            Whether to return a copy with converted matrices
            or to convert the matrices of the existing object.

        Returns
        -------
        ConfusionMatrices
            A collection where the matrices are lists.
            When `copy` is enabled, this is a copy of the original
            collection.
            Allows chaining of methods.
        """
        out = self if not copy else self.__copy__()
        ConfusionMatrices._to_lists(out._matrices, paths=self.paths)
        return out

    def to_ndarrays(self, copy: bool = True):
        """
        Convert all matrices to `numpy.ndarrays`.
        Should only be used after a conversion to lists, as they
        are `numpy.ndarrays` by default.

        Parameters
        ----------
        copy : bool
            Whether to return a copy with converted matrices
            or to convert the matrices of the existing object.

        Returns
        -------
        ConfusionMatrices
            A collection where the matrices are `numpy.ndarray`s.
            When `copy` is enabled, this is a copy of the original
            collection.
            Allows chaining of methods.
        """
        out = self if not copy else self.__copy__()
        ConfusionMatrices._to_ndarrays(out._matrices, paths=self.paths)
        return out

    @staticmethod
    def _to_lists(matrices: dict, paths: List[str]) -> None:
        """
        Convert all arrays to lists.

        Only meaningful when the confusion matrices
        are stored as `numpy.ndarray`s.
        """
        ConfusionMatrices._mutate_matrices(
            matrices=matrices, paths=paths, fn=lambda x: x.tolist()
        )

    @staticmethod
    def _to_ndarrays(matrices: dict, paths: List[str]) -> None:
        """
        Convert all lists to arrays.

        Only meaningful when the confusion matrices
        are stored as lists.
        """
        ConfusionMatrices._mutate_matrices(
            matrices=matrices, paths=paths, fn=np.asarray
        )

    @staticmethod
    def _mutate_matrices(matrices: dict, paths: List[str], fn: Callable) -> None:
        """
        Mutate confusion matrices at the specified paths with a given function.
        """
        for path in paths:
            nested_mutattr(obj=matrices, attr=path, fn=fn)

    def __str__(self) -> str:
        """
        Create string representation of collection
        for printing.
        """
        strings = ["Confusion Matrix Collection:"]
        if self.classes is not None:
            strings += [f"Classes: {', '.join([str(c) for c in self.classes])}"]
        if self.class_roles is not None:
            strings += [f"Class Roles: {_dict_to_str(self.class_roles)}"]
        if self.count_names is not None:
            count_names_string = f"Count Names:\n{self.count_names}"
            count_names_string = re.sub("\n", "\n    ", count_names_string)
            strings += [count_names_string]
        matrices_string = _dict_to_str(self.to_lists(copy=True)._matrices)
        max_matrices_chars = 2000
        if len(matrices_string) > max_matrices_chars:
            matrices_string = matrices_string[:max_matrices_chars] + "..."
        strings += [f"Matrices:\n    {matrices_string}"]
        if self.note:
            strings += [f"Note: {self.note}"]
        return f"\n  ".join(strings)


def _dict_to_str(d):
    """
    Convert dictionary to string.
    """
    # TODO: Consider using https://gist.github.com/jannismain/e96666ca4f059c3e5bc28abb711b5c92
    s = f"{json.dumps(d, sort_keys=True, indent=2)}"
    # Reformat string NOTE: experimental
    s = re.sub(r",[\n\r\s]+", ", ", s)
    s = re.sub(r"\[[\n\r\s]+", "[", s)
    s = re.sub(r"[\n\r\s]+\]", "]", s)
    s = re.sub(r"\n", "\n    ", s)
    return s


def _make_base_int(x):
    """
    Converts numpy integers to base int type.
    Other data types are returned as is.
    """
    if isinstance(x, np.integer):
        return int(x)
    return x
