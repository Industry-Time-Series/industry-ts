"""
    Module with functions for time series feature engineering.
"""
import pandas as pd


def counts_ratio_per_patch(timeseries: pd.Series, patches_dicts: list,
                           column: str,
                           return_timestamps: bool = False) -> pd.DataFrame:
    """
    Calculates the ratio between counts of all values in a batch and the
        samples in the batch. Used for series with discrete inputs as a way of
        aggregating a batch.

    Args:
        timeseries (pd.Series): The time series to be processed

        batches_dicts (list): The list containing the start and end of
            every batch

        column (str): The column name to be used in the ratio calculation

        return_timestamps (bool): If True, the function will return the
            timestamps of the start and end of every batch. Defaults to False.

    Returns:
        df_column_values (pd.DataFrame): A dataframe contaning the ratio of
            column in every data window.
    """

    # Possible values for that column in any batch
    values = list(timeseries.unique())

    # The final list with every ratio value for every batch
    column_values = []

    for event in patches_dicts:
        # Starting moment of the window
        start = event['start']
        # Ending moment of the window
        end = event['end']

        # Create a key for every possible value in that batch
        batch_values = {value: 0 for value in values}

        # Count of each value in that batch
        values_event = timeseries.loc[start:end].value_counts().to_dict()
        size_event = timeseries.loc[start:end].shape[0]

        # * Since the batch does not necessarily has all the values, we use
        # * it's value counts to update the batches values
        for key, _ in values_event.items():
            batch_values[key] = values_event[key]/size_event

        if return_timestamps:
            batch_values["start"] = start
            batch_values["end"] = end

        column_values.append(batch_values)

    df_column_values = pd.DataFrame(column_values)

    for df_column in df_column_values.columns:
        df_column_values = df_column_values.rename(
            {
                df_column: f'{column}_{df_column}'
            },
            axis='columns')

    return df_column_values
