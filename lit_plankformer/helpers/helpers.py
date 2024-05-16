import torch
def remove_useless_columns(df):
    """ Removes columns with no information from dataframe """
    cols_to_check = df.columns.difference(['npimage'])
    # Identify columns where all values are the same, applying the check only to the specified columns
    informative_cols = df.loc[:, df[cols_to_check].nunique(dropna=False) > 1]
    # Check if 'npimage' was in the original DataFrame and add it back to the results if it was
    if 'npimage' in df.columns:
        informative_cols['npimage'] = df['npimage']
    return informative_cols


def output_results(outpath, finetuned, im_names, labels):
    """
    Output the prediction results to a file.
    Args:
        im_names (list): List of image filenames.
        labels (list): List of predicted labels.
    """
    name2 = 'geo_mean_'
    labels = labels.tolist()
    base_filename = f'{outpath}/Ensemble_models_Plankformer_predictions_{name2}{finetuned}'
    file_path = f'{base_filename}.txt'
    lines = [f'\n{img}------------------ {label}\n' for img, label in zip(im_names, labels)]
    with open(file_path, 'w') as f:
        f.writelines(lines)

def gmean(input_x, dim):
    """
    Compute the geometric mean of the input tensor along the specified dimension.
    Args:
        input_x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the geometric mean.
    Returns:
        torch.Tensor: Geometric mean of the input tensor.
    """
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))