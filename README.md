# Python Synthetic Data Vault

This library is an implementation of "The Synthetic Data Vault" described in the following paper works from the MIT:
- https://ieeexplore.ieee.org/document/7796926/
- https://dspace.mit.edu/handle/1721.1/109616

This library was realized in the context of my 4th year internship of computer science engineering school at Ulster University, Derry, Northern Ireland.

🎙️ **DISCLAIMER: *This implementation may not be exactly conform to the original, and hence behavior can differ.***

⚠️ **WARNING: *The version on master might not works with your data set, see the [Versions section](#versions) for more information***

- [Python Synthetic Data Vault](#python-synthetic-data-vault)
    - [Getting started](#getting-started)
        - [Prerequisites](#prerequisites)
        - [Installing](#installing)
        - [Basic usage](#basic-usage)
        - [Usage of the analysis class by class](#usage-of-the-analysis-class-by-class)
        - [Examples](#examples)
    - [Versions](#versions)

## Getting started


### Prerequisites

* Python 3.6 or higher

### Installing

```sh
git clone https://github.com/Taken0711/SDV-python.git
./install.sh
```

It installs the following dependencies:
* NumPy
* SciPy
* Pandas

### Basic usage

The project is built as a library, so you just need to import it in your python code.

```python
import sdv
```

And then use it by calling the `syn` function.

```python
res = sdv.syn(metadata, data)
```

The signature of `syn` is described below.

```python
sdv.syn(metadata, data, size=1, header=None)
```

This return a table with the generated synthetic data, according to the original SDV methodology.

**Parameters**:
- **`metadata`: *array_like***
    - A 1-D array containing the type of data of each column. The available types are `sdv.INT`, `sdv.FLOAT`, `sdv.CATEGORICAL` and `sdv.DATE`.
- **`data`: *array_like***
    - A 2-D array containing the data to synthesize. Types must respect the types described in the `metadata`.
- **`size`: *integer, optional***
    - An integer specifying the number of rows to generate
- **`header`: *array_like, optional***
    - A 1-D array containing the name of the columns. This is used in logs, to help to track which distribution is used to which column.

**Returns**:
- **`res`: *array_like***
    - The generated data set. The size is defined by the `size` parameter.

**Complete example**:
```python
import csv
import sdv

with open("iris.data", "r") as in_file, open("synthetic_iris.data", "w") as out_file:
    iris_reader = csv.reader(in_file)
    header = next(iris_reader)
    ods = [e for e in iris_reader]

    metadata = [sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.CATEGORICAL]
    sds = sdv.syn(metadata, ods, size=len(ods), header=header)

    iris_writer = csv.writer(out_file)
    iris_writer.writerow(header)
    iris_writer.writerows(sds)
```

### Usage of the analysis class by class

This implementation also include an analysis by class. It offer better results than the basic `syn` function by analyzing every class independently.

It can be called as follows:

```python
res = sdv.syn_by_class(metadata, data, class_column)
```

The signature of `syn` is described below.

```python
sdv.syn_by_class(metadata, data, class_column, size=1, header=None):
```

This return a table with the generated synthetic data, by analyzing class by class.

**Parameters**:
- **`metadata`: *array_like***
    - see `syn`
- **`data`: *array_like***
    - see `syn`
- **`class_column`: *integer***
    - An integer specifying the index of the column with the class values
- **`size`: *integer, optional***
    - see `syn`
- **`header`: *array_like, optional***
    - see `syn`

**Returns**:
- **`res`: *array_like***
    - see `syn`

**Complete example**:
```python
import csv
import sdv

with open("iris.data", "r") as in_file, open("synthetic_iris.data", "w") as out_file:
    iris_reader = csv.reader(in_file)
    header = next(iris_reader)
    ods = [e for e in iris_reader]

    metadata = [sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.FLOAT, sdv.CATEGORICAL]
    sds = sdv.syn_by_class(metadata, ods, 4, size=len(ods), header=header)

    iris_writer = csv.writer(out_file)
    iris_writer.writerow(header)
    iris_writer.writerows(sds)
```

### Examples

The [example folder](/examples) contains usage of the library using multiple data set and both methods.

## Versions

As said in the warning, the version on master isn't working for certain data set (I don't exactly know why, it seems to be a type issue with numpy). Thus, there are two versions available:
- 🐇 [Performance version](https://github.com/Taken0711/SDV-python): Works only on certain sets, but around 20 times faster than the slow version
- 🐢 [Slow version](https://github.com/Taken0711/SDV-python/tree/release/v1): Works for every sets (as far as I know) but slowly