"""
    Test the functions in timefunctions.py
"""
import generation.synthetic as syn
import processing.timefunctions as tf


# Test function get_continuous_patches with a simple case
n_discontinuities = 40
timeseries = syn.discontinuous_timeseries("2023-01-01", "2023-12-31",
                                          "1D", n_discontinuities)
patches = tf.get_continuous_patches(timeseries, "1D")

# Test function counts_ratio_per_batch with a simple case
n_discontinuities = 40
timeseries = syn.discontinuous_timeseries("2023-01-01", "2023-12-31",
                                          "1D", n_discontinuities, True)
patches = tf.get_continuous_patches(timeseries, "1D")
column_values = tf.counts_ratio_per_batch(timeseries, patches, 'categories')
