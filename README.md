numerand
========

More random distributions for NumPy.

Currently the only function implemented is ``numerand.select``, a
function similar to the ``choice`` function of the NumPy ``random``
module.  The function allows for sampling with or without replacement.
The API allows the operation of drawing ``nsample`` items without
replacement to be repeated multiple times in a single call.

The implementation of ``numerand.select`` is currently just a proof of
concept for the API.  It works, but some of the calls are implemented
as Python loops, and so will be slow.

The following examples are from the docstring of ``numerand.select``.

Examples
--------
Make 6 choices with replacement from [10, 20, 30, 40].  This is
equivalent to "Make 1 choice without replacement from [10, 20, 30, 40];
do it six times."

    >>> select([10, 20, 30, 40], size=6)
    array([20, 20, 40, 10, 40, 30])

Choose two items from [10, 20, 30, 40] without replacement.  Do it six
times.

    >>> select([10, 20, 30, 40], nsample=2, size=6)
    array([[40, 10],
           [20, 30],
           [10, 40],
           [30, 10],
           [10, 30],
           [10, 20]])

When `nsample` is an integer, there is always an axis at the end of the
result with length `nsample`, even when `nsample=1`.  For example, the
shape of the array returned in the following call is (2, 3, 1)

    >>> select([10, 20, 30, 40], nsample=1, size=(2, 3))
    array([[[10],
            [30],
            [20]],

           [[10],
            [40],
            [20]]])

When `nsample` is None, it acts like `nsample=1`, but the trivial
dimension is not included.  The shape of the array returned in the
following call is (2, 3).

    >>> select([10, 20, 30, 40], size=(2, 3))
    array([[20, 40, 30],
           [30, 20, 40]])
