#


## UnivariateModel
```python 
UnivariateModel(
   order: int, family: str, bias: bool
)
```


---
Base class for univariate models.


**Args**

* **order** (int) : Order of the model.
* **family** (str) : Family of the model. Defaults to "AR".


**Attributes**

* **order**  : Order of the model.
* **family**  : Family of the model.



**Methods:**


### .forecast
```python
.forecast(
   initial_conditions: Union[pd.DataFrame, np.ndarray, list] = None, horizon: int = 1
)
```

---
Simulate the model forward in time.


**Args**

* **initial_conditions** (ArrayLike) : Initial conditions for the model.
* **horizon** (int) : Number of steps to forecast. Defaults to 1.


**Returns**

* **forecast** (ndarray) : Forecasted values.


----


## AutoRegressive
```python 
AutoRegressive(
   p: int = 1, bias: bool = True
)
```


---
Class for purely autoregressive models of order p.


**Args**

* **p** (int) : Order of the model. Defaults to 1.
* **bias** (bool) : Whether to include a bias term in the model.
    Defaults to True.


**Attributes**

* **p**  : Order of the model.
* **coef**  : Coefficients of the model.



**Methods:**


### .fit
```python
.fit(
   data: Union[pd.DataFrame, np.ndarray]
)
```

---
Fit the model to the data.

----


## MovingAverage
```python 
MovingAverage(
   q: int = 1, bias: bool = True
)
```


---
Class for purely moving average models of order q.


**Args**

* **q** (int) : Order of the model. Defaults to 1.
* **bias** (bool) : Whether to include a bias term in the model.
    Defaults to True.


**Attributes**

* **q**  : Order of the model.
* **coef**  : Coefficients of the model.



**Methods:**


### .fit
```python
.fit(
   data: Union[pd.DataFrame, np.ndarray], n_iterations: int = 100
)
```

---
Fit the model to the data.
