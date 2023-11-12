#


## LeastSquaresOptimizer
```python 
LeastSquaresOptimizer(
   method: str = 'OLS'
)
```


---
Class with different least squares methods.


**Args**

* **method** (str) : Method to use. Is one of the following:
    'OLS': Ordinary least squares.
    'RLS': Recursive least squares.
    'ELS': Extended least squares. (Not implemented yet)
    'regOLS': Regularized least squares. (Not implemented yet)
  Defaults to "OLS".


**Methods:**


### .fit
```python
.fit(
   regressors: np.ndarray, targets: np.ndarray, inplace: bool = False, **kwargs
)
```

---
Fit the model to the data.


**Args**

* **regressors** (ndarray) : Matrix with regressors, commonly denominated
    the Phi matrix. Each column is a regressor, and each row is an
    observation.
* **targets** (ndarray) : Vector with targets, commonly denominated the
    y vector. Each row is an observation.
* **kwargs**  : Additional keyword arguments for the different methods.


**Returns**

* **coef** (ndarray) : Coefficients of the model.

