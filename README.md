numerand
========

More random distributions for NumPy.

``numerand`` has functions for:

* [generating random contingency tables](#Random-contingency-tables)
* [selecting random elements from a sequence, with
  or without replacement](#Random-selection-of-elements-from-a-sequence)

# Random contingency tables

`numerand` has the functions `random_table()` and `random_table_from_table()`.
Both generate random contingency tables using the same underlying method.
The only difference is in the inputs to the functions.

The Python function `random_table_from_table(table, size=None, rng=None)`
generates a random contingency table with the same variable sums as the
given table.

The function `random_table(*sums, size=None, rng=None)` accepts the
quantities associated with the levels of each categorical variable
(i.e. the marginal sums for each variable).

*Note.* For contingency tables larger than 2x2, the method used to generate
a random table is simple but slow.  Let ``m`` be the number of dimensions
of ``table``, and let ``N`` be the sum of the elements in the table.  Both
the time and space complexities of the algorithm to generate one sample are
O(``m*N``). There is a lot of literature on the efficient generation of
random contingency tables that has been completely ignored here!

Examples
--------

```
>>> import numpy as np
>>> from numerand import random_table_from_table, random_table

>>> c = np.array([[36, 5, 14], [14, 10, 21]])
>>> c 
array([[36,  5, 14],
       [14, 10, 21]])
```

Generate one random table:

```
>>> random_table_from_table(c) 
array([[28,  6, 21],
       [22,  9, 14]])

```

The `size` parameter allows several random tables to be generated in one call:

```
>>> random_table_from_table(c, size=3) 
array([[[28,  7, 20],
        [22,  8, 15]],

       [[27, 10, 18],
        [23,  5, 17]],

       [[26,  7, 22],
        [24,  8, 13]]])
```

Generate a large number of tables, and compare the sample mean to the expected
mean computed by `scipy.stats.contingency.expected_freq`:

```
>>> tables = random_table_from_table(c, size=100000)

>>> tables.mean(axis=0) 
array([[27.49068,  8.24867, 19.26065],
       [22.50932,  6.75133, 15.73935]])

>>> from scipy.stats.contingency import expected_freq

>>> expected_freq(c) 
array([[27.5 ,  8.25, 19.25],
       [22.5 ,  6.75, 15.75]])
```

Higher dimensional tables are accepted.  Here, `c3` has shape `(2, 3, 3)`.
We use `scipy.stats.contingency.margins` to compute the marginal sums
for `c3`.

```
>>> c3 = np.array([[[24, 15, 9], [26, 15, 6], [5, 11, 14]],
                   [[40, 11, 7], [21, 10, 12], [9, 8, 7]]])
>>> c3
array([[[24, 15,  9],
        [26, 15,  6],
        [ 5, 11, 14]],

       [[40, 11,  7],
        [21, 10, 12],
        [ 9,  8,  7]]])

>>> from scipy.stats.contingency import margins

>>> margins(c3)
[array([[[125]],

        [[125]]]),
 array([[[106],
         [ 90],
         [ 54]]]),
 array([[[125,  70,  55]]])]
```
Generate a random table from `c3`.
```
>>> sample = random_table_from_table(c3)

>>> sample
array([[[20, 15, 10],
        [21, 14, 12],
        [22,  6,  5]],

       [[28, 16, 17],
        [21, 14,  8],
        [13,  5,  3]]])
```
Verify that the marginal sums of `sample` are the same as those of `c3`:
```
>>> margins(sample)
[array([[[125]],

        [[125]]]),
 array([[[106],
         [ 90],
         [ 54]]]),
 array([[[125,  70,  55]]])]
```

Instead of an existing contingency table, the function `random_table()`
accepts the marginal sums associated with each categorical variable.
For example,

```
>>> random_table([10, 10, 20], [15, 25])
array([[ 3,  7],
       [ 4,  6],
       [ 8, 12]])
```
Note that the sums of the rows are [10, 10, 20], and the sums of
the columns are [15, 25].

Higher-dimensional tables can also be generated with this function.
The following generates one contingency table with shape (3, 2, 3):
```
>>> table = random_table([10, 10, 20], [15, 25], [6, 28, 6])
>>> table
array([[[0, 3, 0],
        [0, 6, 1]],

       [[1, 3, 0],
        [0, 5, 1]],

       [[2, 4, 2],
        [3, 7, 2]]])
```
Verify that the margins have the required sums.

```
>>> [t.ravel() for t in margins(table)]
[array([10, 10, 20]), array([15, 25]), array([ 6, 28,  6])]
```

# Random selection of elements from a sequence

``numerand.select`` is a function similar to the ``choice`` function
of the NumPy ``random`` module.  The function allows for sampling with
or without replacement.  The API allows the operation of drawing ``nsample``
items without replacement to be repeated multiple times in a single call.

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
