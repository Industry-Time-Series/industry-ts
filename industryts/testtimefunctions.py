"""
    Test the functions in timefunctions.py
"""
import generation.synthetic as syn
import processing.timefunctions as tf
import pandas as pd

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
patches = tf.get_continuous_patches(timeseries, frequency)

# Test function counts_ratio_per_patch ----------------------------------------
n_discontinuities = 40
timeseries = syn.discontinuous_timeseries(start_date, end_date,
                                          frequency, n_discontinuities, True)
patches = tf.get_continuous_patches(timeseries, frequency)
column_values = tf.counts_ratio_per_patch(timeseries, patches, 'categories')

# Test function rm_stopped_operation ------------------------------------------
# No discontinuities
n_discontinuities = 0
timeseries_a = syn.discontinuous_timeseries(start_date, end_date,
                                            frequency, n_discontinuities)
timeseries_b = syn.discontinuous_timeseries(start_date, end_date,
                                            frequency, n_discontinuities)
dataframe = pd.DataFrame({'a': timeseries_a, 'b': timeseries_b})
print("Original shape", dataframe.shape)
print("Rows to be removed", dataframe[dataframe['a'] >= 0.8].shape)
# Suppose that we want to remove the batches where a >= 0.8
mask = dataframe['a'] >= 0.8
# Removing only the rows where a >= 0.8
dataframe_0_0 = tf.rm_stopped_operation(dataframe, mask,
                                        return_shutdown_dict=False)
print("Shape after removing only stopped operation", dataframe_0_0.shape)

# Test remove 2D before and after
dataframe_2_2 = tf.rm_stopped_operation(dataframe, mask, "2D", "2D",
                                        return_shutdown_dict=False)
print("Shape after removing 2 days before and after", dataframe_2_2.shape)
patches = tf.get_continuous_patches(timeseries, frequency)
print("Number of patches", len(patches))
