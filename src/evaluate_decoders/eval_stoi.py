import pandas as pd
import soundfile as sf
import numpy as np
from pystoi import stoi

def calculate_stoi(prediction_path, target_path, pad=False):
    """
    input: target and prediction audio files; ESTOI = extended version of STOI
    output: pandas DF with stoi values
    """

    target_wav, fs = sf.read(target_path)
    prediction_wav, fs = sf.read(prediction_path)

    # Pad if one is shorter
    if pad:
        if target_wav.shape[0] < prediction_wav.shape[0]:
            target_wav = np.pad(target_wav, [0, prediction_wav.shape[0] - target_wav.shape[0]], constant_values=0)
        elif prediction_wav.shape[0] < target_wav.shape[0]:
            prediction_wav = np.pad(prediction_wav, [0, target_wav.shape[0] - prediction_wav.shape[0]], constant_values=0)

    # Clean and den should have the same length, and be 1D
    d = stoi(target_wav, prediction_wav, fs, extended=False)
    ed = stoi(target_wav, prediction_wav, fs, extended=True)
    return pd.DataFrame({'stoi': [d], 'estoi': [ed]})