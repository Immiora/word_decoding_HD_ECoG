import pandas as pd
import numpy as np

def calculate_pearsonr(predictions, targets):
    """
    input: 3d arrays: trials x time x mel features
    output: pandas DF with correlation values per mel feature
    """

    res_pearsonr = {'r': []}
    for p, t in zip(predictions, targets):
        res_pearsonr['r'] = [np.corrcoef(ip, it)[0, 1] for ip, it in zip(p.T, t.T)]
    return pd.DataFrame(res_pearsonr)


def calculate_pearsonr_flattened(predictions, targets):
    """
    input: 3d arrays: trials x time x mel features
    output: pandas DF with 1 correlation value

    correlate flattened spectrograms (time x freq)
    """
    assert predictions.ndim == 3 and targets.ndim == 3, 'Inputs need to be 3d'
    res_pearsonr = []
    for p, t in zip(predictions, targets):
        res_pearsonr.append(np.corrcoef(p.flatten(), t.flatten())[0, 1])
    return pd.DataFrame({'r':res_pearsonr})