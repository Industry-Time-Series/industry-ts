"""
    Module with functions for time series feature engineering.
"""
import pandas as pd


def counts_ratio_per_patch(timeseries: pd.Series, patches_dicts: list,
                           column: str,
                           return_timestamps: bool = False) -> pd.DataFrame:
    """
    Calculates the ratio between counts of all values in a patch of data
        Used for series with discrete inputs for data aggregation.

    This function expects a list of dictionaries with the start and end of
        every patch, in the same format as the output of the function
        `industryts.processing.filtering.get_patches_dicts`.

    Each item in the patches_dicts list is a dictionary with the following
        structure:
        {
            'start': start_timestamp,
            'end': end_timestamp
        }

    The function returns a dataframe in which each column stores the ratio
        of a value in the column parameter in every patch. The column names
        follow the convention "category_originalColName".

    Args:
        timeseries (pd.Series): The time series to be processed

        patches_dicts (list): The list containing the start and end of
            every data patch

        column (str): The column name to be used in the ratio calculation

        return_timestamps (bool): If True, the function will return the
            timestamps of the start and end of every patch. Defaults to False.

    Returns:
        df_column_values (pd.DataFrame): A dataframe contaning the ratio of
            column in every data window.
    """

    # Possible values for that column in any patch
    values = list(timeseries.unique())

    # The final list with every ratio value for every patch
    column_values = []

    for event in patches_dicts:
        # Starting moment of the window
        start = event['start']
        # Ending moment of the window
        end = event['end']

        # Create a key for every possible value in that patch
        patch_values = {value: 0 for value in values}

        # Count of each value in that patch
        values_event = timeseries.loc[start:end].value_counts().to_dict()
        size_event = timeseries.loc[start:end].shape[0]

        # * Since the patch does not necessarily has all the values, we use
        # * its value counts to update the patches values
        for key, _ in values_event.items():
            patch_values[key] = values_event[key]/size_event

        if return_timestamps:
            patch_values["start"] = start
            patch_values["end"] = end

        column_values.append(patch_values)

    df_column_values = pd.DataFrame(column_values)

    for df_column in df_column_values.columns:
        df_column_values = df_column_values.rename(
            {
                df_column: f'{column}_{df_column}'
            },
            axis='columns')

    return df_column_values
