from lidartouch.utils.types import is_list

def make_list(var, n=None):
    """
    Wraps the input into a list, and optionally repeats it to be size n
    Parameters
    ----------
    var : Any
        Variable to be wrapped in a list
    n : int
        How much the wrapped variable will be repeated
    Returns
    -------
    var_list : list
        List generated from var
    """
    var = var if is_list(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list'
        return var * n if len(var) == 1 else var

def same_shape(shape1, shape2):
    """
    Checks if two shapes are the same
    Parameters
    ----------
    shape1 : tuple
        First shape
    shape2 : tuple
        Second shape
    Returns
    -------
    flag : bool
        True if both shapes are the same (same length and dimensions)
    """
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False
    return True