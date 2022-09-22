import torch

def average_metrics(list_of_metrics, prefix):
    """
    Average metrics values in a list of batches

    Parameters
    ----------
    list_of_metrics : list of dict
        List containing dictionaries with the same keys across dicts

        e.g.:
        [
            {
            'photometric_loss': 0.5, # each value has been averaged over its batch
            'smoothness_loss': 0.8
            },
            {
            'photometric_loss': 0.9,
            'smoothness_loss': 0.6
            }
        ]

    batch_size: int
        batch_size used to computes these metrics

    prefix : str
        Prefix string for metrics logging

    Returns
    -------
    avg_values : dict
        Dictionary containing keys of the dicts and the averaged values as value

        {
            'prefix_photometric_loss': 0.5, # each value has been averaged over the total number of elements
            'prefix_smoothness_loss': 0.8
        }
    """
    avg_values = {}

    for metric_key in list_of_metrics[0].keys():
        avg_value = torch.stack([metrics[metric_key] for metrics in list_of_metrics]).mean()
        avg_values[f"{prefix}{metric_key}"] = avg_value

    return avg_values