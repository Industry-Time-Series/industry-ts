#


### infer_sampling_time
```python
.infer_sampling_time(
   data: pd.DataFrame
)
```

---
Infers sampling time from datetime index.
If the index is not a datetime index, it will raise an error.
Returns the sampling time as a pd.Timedelta object.

The algorithm works by taking the first 10 samples and inferring
the frequency from them. If the frequency is not inferred, it will
take the next 10 samples and try again. This is repeated until the
frequency is inferred or all samples are exhausted.


**Args**

* **data** (pd.DataFrame) : Data frame with datetime index to infer
  the sampling time


**Returns**

* **t_s** (pd.Timedelta) : The inferred sampling time

