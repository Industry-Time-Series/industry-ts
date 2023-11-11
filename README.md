# Industry Time Series Library

## Time Series Manipulation for Industry Data

This repository hosts the **industry-ts**: _Industry Time Series_ Library --- a Python library that provides functions to manipulate time series collected from industrial contexts.

This project aims to address the necessity for open-source tools developed for solving problems with data collected from industrial process. 

The modules introduced provide a variety of functions that are particularly tailored to industry data, directed to working with its most common issues, such as discontinuities in process measurements and faulty sensors.

## Table of Contents

- [Industry Time Series Library](#industry-time-series-library)
  - [Time Series Manipulation for Industry Data](#time-series-manipulation-for-industry-data)
  - [Table of Contents](#table-of-contents)
  - [Main Features](#main-features)
  - [How to use it](#how-to-use-it)
  - [Documentation](#documentation)

## Main Features

* **Data Generation**: Generate synthetic data from well defined stochastic processes for testing and benchmarking purposes.
* **Modelling**: Fit time series models to data.
* **Preprocessing**: Preprocess time series with filtering, feature engineering and other techniques.

## How to use it

The package is available in [PyPI](https://pypi.org/project/industryts/), and can be installed with pip:

```bash
pip install industryts
```

Alternatively, to use the library, you can clone the repository and install it with pip:

```bash
git clone https://github.com/Industry-Time-Series/industry-ts.git
cd industry-ts
git checkout packaging
pip install .
```

## Documentation

The official documentation is hosted on https://industry-time-series.github.io/industry-ts/
