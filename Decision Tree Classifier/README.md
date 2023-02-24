# CS311 Programming Assignment 3

For this assignment, you will be training and testing Decisions Trees. Refer to the Canvas assignment for assignment specifications. This README describes how to run the skeleton code.

## Running the skeleton program

The skeleton code benchmarks your search functions on a set of game boards. Executing `csp.py` will run backtracking search with AC3 on an "easy" board by default.  You can change the algorithm, the difficulty level, the number of trials and specify the board by changing the optional arguments shown below.

```
$ python3 decision.py -h
usage: decision.py [-h] [-p PREFIX] [-k K_SPLITS]

Train and test decision tree learner

optional arguments:
  -h, --help            show this help message and exit
  -p PREFIX, --prefix PREFIX
                        Prefix for dataset files. Expects <prefix>.[train|test]_[data|label].txt files (except for adult). Allowed values: small1, hepatitis, adult.
  -k K_SPLITS, --k_splits K_SPLITS
                        Number of splits for stratified k-fold testing
```

For example, to the other datasets, run the program as `python3 csp.py -p adult`.

If you are working with Thonny, recall that you can change the command line arguments by modifying the `%Run` command in the shell, e.g., `%Run decision.py -p adult`.

## Unit testing

To assist you during development, a unit test suite is provided in `decision_test.py`. These tests are a subset of the tests run by Gradescope. You can run the tests by executing the `decision_test.py` file as a program, e.g. `python3 decision_test.py`. 

```
$ python3 decision_test.py
......
----------------------------------------------------------------------
Ran 6 tests in 0.142s

OK
```
