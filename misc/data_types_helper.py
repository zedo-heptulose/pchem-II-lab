"""
Helper functions for the Data Types mini-assignment.
"""

import numpy as np


# Map of accepted guess strings to actual Python type names
_TYPE_ALIASES = {
    "list": "list",
    "ndarray": "ndarray",
    "array": "ndarray",
    "numpy array": "ndarray",
    "int": "int",
    "float": "float",
    "str": "str",
    "string": "str",
    "bool": "bool",
    "boolean": "bool",
}


def _get_type_label(obj):
    """Return a short, student-friendly type label for *obj*.

    Parameters
    ----------
    obj : object
        Any Python object.

    Returns
    -------
    str
        One of 'list', 'ndarray', 'int', 'float', 'str', 'bool',
        or the raw class name for unexpected types.
    """
    if isinstance(obj, np.ndarray):
        return "ndarray"
    return type(obj).__name__


def check_guess(expression, guess):
    """Evaluate *expression* and compare its type to the student's *guess*.

    Parameters
    ----------
    expression : callable
        A zero-argument callable (typically a lambda) whose return value
        will be inspected.
    guess : str
        The student's predicted type name (e.g. ``'list'``, ``'ndarray'``).
    """
    result = expression()
    actual = _get_type_label(result)
    normalized = _TYPE_ALIASES.get(guess.strip().lower(), guess.strip().lower())

    print(f"Expression result : {result!r}")
    print(f"Actual type       : {actual}")
    print()

    if normalized == actual:
        print("Correct!")
    else:
        print(f"Not quite — you guessed '{guess.strip()}', but the actual type is '{actual}'.")
        print()
        # Provide short explanations for common surprises
        if actual == "list" and normalized in ("ndarray",):
            print("Hint: The `*` operator on a plain Python list repeats its elements,")
            print("it does not perform element-wise multiplication.")
        elif actual == "ndarray" and normalized in ("list",):
            print("Hint: When a NumPy array appears in an arithmetic operation,")
            print("the result is also a NumPy array (element-wise math).")
        elif actual == "str" and normalized in ("int", "float"):
            print("Hint: The `*` operator on a string repeats the string,")
            print("it does not perform numerical multiplication.")
        elif actual == "int" and normalized in ("float",):
            print("Hint: Multiplying two ints in Python gives an int.")
            print("You only get a float when at least one operand is a float.")
