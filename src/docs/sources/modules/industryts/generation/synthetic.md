#


### ar_process
```python
.ar_process(
   coefs: list, samples: int = 100, noise: float = 0
)
```

---
Generate synthetic data from an Autoregressive (AR) process of a given
length and known coefficients, with the possibility of adding noise to
the measurements.


**Args**

* **coefs** (list) : list with coefficients of lagged measurements of
* **samples** (int) : number of data points to be generated. Default is 100.
* **noise** (float) : standard deviation of the noise to be added to the
  the series. The order of the AR process will be defined by the number
  of defined coefficients. For example, if coefs = [0.5, 0.3], the
  generated series will be an AR(2) process, where 0.5 is the
  coefficient of the first lagged measurement and 0.3 is the
  coefficient of the second lagged measurement.
measurements. Default is 0, which means no noise.


**Returns**

* **series**  : array with the generated AR process.


----


### ma_process
```python
.ma_process(
   coefs: list, samples: int = 100, noise: float = 0
)
```

---
Generate synthetic data from a Moving Average (MA) process of a given
length and known coefficients, with the possibility of adding noise to
the measurements.


**Args**

* **coefs** (list) : list with coefficients of lagged measurements of
* **samples** (int) : number of data points to be generated. Default is 100.
* **noise** (float) : standard deviation of the noise to be added to the
  the series. The order of the MA process will be defined by the number
  of defined coefficients. For example, if coefs = [0.5, 0.3], the
  generated series will be an MA(2) process, where 0.5 is the
  coefficient of the first lagged measurement and 0.3 is the
  coefficient of the second lagged measurement.
  measurements. Default is 0, which means no noise.


**Returns**

* **series**  : array with the generated MA process.


----


### seasonal_component
```python
.seasonal_component(
   samples: int = 100, period: int = 10, amplitude: float = 1, noise: float = 0
)
```

---
Generate a seasonal component of a given length, period and amplitude,
with the possibility of adding noise to the measurements.

The nature of the seasonal component is a sine wave with the given period
and amplitude.


**Args**

* **samples** (int) : number of data points to be generated. Default is 100.
* **period** (int) : period of the seasonal component. Default is 10.
* **amplitude** (float) : amplitude of the seasonal component. Default is 1.
* **noise** (float) : standard deviation of the noise to be added to the
measurements. Default is 0, which means no noise.


**Returns**

* **series**  : array with the generated seasonal component.


----


### trend_component
```python
.trend_component(
   samples: int = 100, slope: float = 0.1, intercept: float = 0, noise: float = 0
)
```

---
Generate a trend component of a given length, slope and intercept, with
the possibility of adding noise to the measurements.


**Args**

* **samples** (int) : number of data points to be generated. Default is 100.
* **slope** (float) : slope of the trend component. Default is 0.
* **intercept** (float) : intercept of the trend component. Default is 0.
* **noise** (float) : standard deviation of the noise to be added to the
measurements. Default is 0, which means no noise.


**Returns**

* **series**  : array with the generated trend component.


----


### discontinuous_timeseries
```python
.discontinuous_timeseries(
   start_timestamp: Union[str, pd.Timestamp], end_timestamp: Union[str,
   pd.Timestamp], freq: Union[str, pd.Timedelta], num_discontinuities: int,
   is_categorical: bool = False
)
```

---
Generate a random discontinuous time series with given start and end
timestamps and frequency.


**Args**

* **start_timestamp** (str or Timestamp) : start date of the time series.
* **end_timestamp** (str or Timestamp) : end date of the time series.
* **freq** (str or Timedelta) : frequency of the time series.
* **num_discontinuities** (int) : number of discontinuity points to be
* **is_categorical** (bool) : if True, the time series will be categorical
  generated.
  with levels 'A', 'B', 'C' and 'D'. Default is False.


**Returns**

* **ts** (Series) : discontinuous time series with random data.


----


### part_static_timeseries
```python
.part_static_timeseries(
   start_timestamp: Union[str, pd.Timestamp], end_timestamp: Union[str,
   pd.Timestamp], frequency: Union[str, pd.Timedelta], n_samples_static: int,
   value_static: float = 1.0
)
```

---
Generate a time series with a part that is static.


**Args**

* **start_timestamp** (Union[str, pd.Timestamp]) : Start date of the time
* **end_timestamp** (Union[str, pd.Timestamp]) : End date of the time series.
* **n_samples_static** (int) : Number of samples that will be static.
  series.


**Raises**

* **ValueError**  : Error raised if the number of samples is less than the
  number of static samples.


**Returns**

* **DataFrame**  : Time series with a part that is static.

