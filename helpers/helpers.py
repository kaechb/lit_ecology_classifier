
def remove_useless_columns(df):
    """ Removes columns with no information from dataframe """
    cols_to_check = df.columns.difference(['npimage'])
    # Identify columns where all values are the same, applying the check only to the specified columns
    informative_cols = df.loc[:, df[cols_to_check].nunique(dropna=False) > 1]
    # Check if 'npimage' was in the original DataFrame and add it back to the results if it was
    if 'npimage' in df.columns:
        informative_cols['npimage'] = df['npimage']
    return informative_cols


