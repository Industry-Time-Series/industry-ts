#


### get_continuous_patches
```python
.get_continuous_patches(
   data: Union[pd.DataFrame, np.ndarray], sampling_time: pd.Timedelta = None,
   min_length: Union[str, int] = 0, drop_leading: int = 0, drop_trailing: int = 0,
   return_num_index: bool = False
)
```

---
Function to extract the time limits for all continuous patches of data.

In the context of industrial processes, each continuous patch may be
associated with a batch, or simply a period of time where the process is
continuously active between shutdowns.

This function is often used in conjunction with a function for removivng
samples in the data that were collected in periods of shutdown.


**Args**

* **data** (pd.DataFrame) : The data to obtain the start and end timestamps.
* **sampling_time** (pd.Timedelta) : Sampling time. If None, it will be
    inferred from the data index. Defaults to None.
* **min_length** (str, int) : The minimum duration to consider a patch as
* **drop_leading** (int, optional) : Number of samples to drop from
    the start of each patch of data. Defaults to 0.
* **drop_trailing** (int, optional) : Number of samples to drop from the end
    of each patch of data. Defaults to 0.
* **return_num_index** (bool, optional) : If True, the function will return
    the row number of the start and end of each patch,
    instead of the timestamp index (if available).


valid. If 'str', it must be a string that can be converted to a
pd.Timedelta. If 'int', it must be an integer representing the number
of samples. Defaults to 0, which means that all patches will be
considered.




**Returns**

* **patches** (list) : The list of dicts containing the start and end
    of each patch.


----


### rm_stopped_operation
```python
.rm_stopped_operation(
   data: pd.DataFrame, rm_events_mask: np.ndarray, rm_interval_start: Union[str,
   pd.Timedelta] = '0S', rm_interval_stop: Union[str, pd.Timedelta] = '0S',
   minimum_interval: Union[str, pd.Timedelta] = '0S',
   return_shutdown_dict: bool = False
)
```

---
Remove all samples in rm_events_mask, plus/minus stop_interval.


**Args**

* **data** (pd.DataFrame) : data to be processed, must have datetime indexing
* **rm_events_mask** (ndarray) : boolean ndarray with length equal to number
                     of rows in data, where rows to be removed are True
* **rm_interval_start** (Union[str, pd.Timedelta]) : time interval to be
                        removed before the events in the mask
* **rm_interval_stop** (Union[str, pd.Timedelta]) : time interval to be
                        removed after the events in the mask
* **minimum_interval** (string) : mininum duration of a stop
* **return_shutdown_dict** (bool) : if True, returns a dictionary with start
                            and end indices of all events







**Returns**

* **DataFrame**  : data with the rows of rm_events_mask and samples around
    stop_interval removed, in addition to (optionally) shutdown dict.


----


### filter_static_windows
```python
.filter_static_windows(
   data: pd.DataFrame, columns: list = None, threshold: int = 10,
   remove_window: bool = False, return_n_removed: bool = True
)
```

---
Filter windows where there is no variation for 'threshold' consecutive
samples. In other words, replace static windows for null values or
remove windows, where a window is considered static if the value of the
series remains unchanged for at least 'threshold' samples.


**Args**

* **data** (pd.DataFrame) : Dataframe with the values to be analyzed
* **columns** (list, optional) : Columns to check if the windows are static.
    Defaults to [].
* **threshold** (int, optional) : Minimum length of sequence of samples with
    the same value to remove. Defaults to 10.
* **remove_window** (bool, optional) : If True, removes the static windows.
    If False, replace static windows with NA. Defaults to False.
* **return_n_removed** (bool, optional) : If True, returns the amount of
    removed samples for every column. Defaults to True.


**Returns**

* **DataFrame**  : Dataframe without the static windows (or NAs inplace).
* **DataFrame**  : Columns with amount of static windows removed (or NAs
inplace).

----


### remove_static_columns
```python
.remove_static_columns(
   df: pd.DataFrame, min_std_cv: float = 0.01, columns: list = None,
   return_removed: bool = True
)
```

---
Removes columns for which the coefficient of variation is below a selected
threshold.

If the mean of the column is 0, the standard deviation is used instead
of the coefficient of variation.


**Args**

* **df** (pd.DataFrame) : Process dataframe
* **min_std_cv** (float, optional) : Minimum variation coefficient.
    Defaults to 0.01.
* **columns** (list, optional) : List of columns to check. If None, all
    columns are checked. Defaults to None.
* **return_removed** (bool, optional) : If True, returns the removed columns
    and their coefficient of variation. Defaults to True.


**Returns**

* **DataFrame**  : The dataframe without the columns that don't change
* **DataFrame**  : Dataframe with the columns that don't change and their
    coefficient of variation


----


### format_start
```python
.format_start(
   df: pd.DataFrame, s: int = 0, m: int = 0, h: int = 0
)
```

---
Function to remove (if necessary) the first rows of the data in
order to have the first row in a specific format.

---
If the formatting is not possible, the function returns the original
    dataframe.


**Args**

* **df** (pd.DataFrame) : df whose index is in datetime format.
* **s** (int) : initial second, with all windows starting with this second.
* **m** (int) : initial minute, with all windows starting with this minute.
* **h** (int) : if 0, initial hours are even; if 1, initial hours are odd.



**Returns**

* **DataFrame**  : with the specified format.

