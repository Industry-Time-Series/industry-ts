"""
    Test the functions in timefunctions.py
"""
import generation.synthetic as syn
import processing.timestamps as tf
import processing.filtering as flt
import pandas as pd
import processing.featureengineering as fe


"""The expected flow of work is the following:
1. Generate synthetic data without discontinuities
2. Remove some data points to simulate discontinuities (e.g. shutdowns).
In real life we have in fact shutdowns.
3. Get the continuous patches of the time series
4. Compute the ratio of counts per patch in case there are categories
"""

start_date = "2023-01-01"
end_date = "2023-12-31"
frequency = "1D"

# Test function get_continuous_patches ----------------------------------------
n_discontinuities = 40
timeseries = syn.discontinuous_timeseries(start_date, end_date,
                                          frequency, n_discontinuities)
patches = flt.get_continuous_patches(timeseries, frequency)

# Test function counts_ratio_per_patch ----------------------------------------
n_discontinuities = 40
timeseries = syn.discontinuous_timeseries(start_date, end_date,
                                          frequency, n_discontinuities, True)
patches = flt.get_continuous_patches(timeseries, frequency)
column_values = fe.counts_ratio_per_patch(timeseries, patches, 'categories')

# Test function rm_stopped_operation ------------------------------------------
# No discontinuities
n_discontinuities = 0
timeseries_a = syn.discontinuous_timeseries(start_date, end_date,
                                            frequency, n_discontinuities)
timeseries_b = syn.discontinuous_timeseries(start_date, end_date,
                                            frequency, n_discontinuities)
dataframe = pd.DataFrame({'a': timeseries_a, 'b': timeseries_b})
print("=====================================================")
print("Test rm_stopped_operation")
print("Original shape", dataframe.shape)
print("Rows to be removed", dataframe[dataframe['a'] >= 0.8].shape)
# Suppose that we want to remove the batches where a >= 0.8
mask = dataframe['a'] >= 0.8
# Removing only the rows where a >= 0.8
dataframe_0_0 = flt.rm_stopped_operation(dataframe, mask,
                                        return_shutdown_dict=False)
print("Shape after removing only stopped operation", dataframe_0_0.shape)

# Test remove 2D before and after
dataframe_2_2 = flt.rm_stopped_operation(dataframe, mask, "2D", "2D",
                                        return_shutdown_dict=False)
print("Shape after removing 2 days before and after", dataframe_2_2.shape)
patches = flt.get_continuous_patches(timeseries, frequency)
print("Number of patches", len(patches))
print("=====================================================")
# Test function filter_static_windows =========================================
n_samples_static = 100
frequency = "1 minute"
print("Test filter_static_windows")
# Replace the static part with NaNs
timeseries_static = syn.part_static_timeseries(
    start_date, end_date, frequency, n_samples_static)
print("NAs in the static part", timeseries_static.isna().sum())
timeseries_static, removed_df = flt.filter_static_windows(
    timeseries_static, threshold=10)
print("Shape after filtering static part", timeseries_static.shape)
print("NAs after filtering the static part", timeseries_static.isna().sum())
print("Removed dataframe", removed_df)

# Removing in fact the static part
timeseries_static = syn.part_static_timeseries(
    start_date, end_date, frequency, n_samples_static)
print("Original shape", timeseries_static.shape)
timeseries_static, removed_df = flt.filter_static_windows(
    timeseries_static, threshold=10, remove_window=True)
print("Shape after removing static part", timeseries_static.shape)
print("Removed dataframe", removed_df)
print("=====================================================")
# Test function remove_static_columns and remove_almost_static_columns ========
start_date = "2023-01-01"
end_date = "2023-12-31"
frequency = "1D"
n_samples_static = 365
print("Test remove_static_columns and remove_almost_static_columns")
timeseries_static = syn.part_static_timeseries(
    start_date, end_date, frequency, n_samples_static, 1000)
print("Original shape", timeseries_static.shape)
timeseries_static, removed_df = flt.remove_static_columns(timeseries_static)
print("Shape after removing static columns", timeseries_static.shape)
print("Removed dataframe", removed_df)

n_samples_static = 330
timeseries_static = syn.part_static_timeseries(
    start_date, end_date, frequency, n_samples_static, 1000)
print("Original shape", timeseries_static.shape)
timeseries_static = flt.remove_almost_static_columns(timeseries_static)
print("Shape after removing almost static columns", timeseries_static.shape)
print("=====================================================")
# Test function format_start ==================================================
start_date = "2023-01-01"
end_date = "2023-01-31"
frequency = "60S"
print("Test format_start")
n_discontinuities = 100
timeseries_a = syn.discontinuous_timeseries(start_date, end_date,
                                            frequency, n_discontinuities)
patches = flt.get_continuous_patches(timeseries_a, frequency)

# For every patch, we need that it starts at minute 1 at a odd hour
patch = patches[0]
print("Patch", patch)
timeseries_b = timeseries_a[patch["start"]:patch["end"]]
print("Start before format_start", timeseries_b.index[0])
timeseries_c = flt.format_start(timeseries_b, 0, 1, 1)
print("Start after format_start", timeseries_c.index[0])
print("=====================================================")
