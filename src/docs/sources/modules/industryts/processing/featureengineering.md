#


### counts_ratio_per_patch
```python
.counts_ratio_per_patch(
   timeseries: pd.Series, patches_dicts: list, column: str,
   return_timestamps: bool = False
)
```

---
Calculates the ratio between counts of all values in a patch of data
Used for series with discrete inputs for data aggregation.

---
This function expects a list of dictionaries with the start and end of
    every patch, in the same format as the output of the function
    `industryts.processing.filtering.get_patches_dicts`.

Each item in the patches_dicts list is a dictionary with the following
    structure:
    {
    }

The function returns a dataframe in which each column stores the ratio
    of a value in the column parameter in every patch. The column names
    follow the convention "category_originalColName".


**Args**

* **timeseries** (pd.Series) : The time series to be processed
* **patches_dicts** (list) : The list containing the start and end of
    every data patch
* **column** (str) : The column name to be used in the ratio calculation
* **return_timestamps** (bool) : If True, the function will return the
    timestamps of the start and end of every patch. Defaults to False.





**Returns**

* **df_column_values** (pd.DataFrame) : A dataframe contaning the ratio of
    column in every data window.

