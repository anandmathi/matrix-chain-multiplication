# Matrix Chain Multiplication

See the Report for an in-depth analysis of the algorithm.

## Usage

To run with an input file, use the command:

 `python3 MatrixChainMultiplication.py <input-file>`

You can use the provided input.txt file if you don't want to create your own.

To run unit and measurement tests, use the command:

`python3 -m unittest test_MatrixChainMultiplication.py`

The input file must be formatted as follows:

```
1, 1, 1
1, 1, 1
---
1, 1
1, 1
1, 1
---
1, 1, 1
1, 1, 1
---
...
```

where the rows and columns of each block corresponds with the elements of a matrix. Separate matrices with `---`. Ensure that the columns of matrix `i` is equal to the rows of matrix `i+1` or it will throw an error (because it is not compatible with multiplication).