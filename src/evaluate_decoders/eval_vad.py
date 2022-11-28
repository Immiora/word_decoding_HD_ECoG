import pandas as pd
import numpy as np
import subprocess
import re


def run_vad(fname, aggressiveness):
    out = subprocess.run(['python',
                          'evaluate_decoders/vad.py',
                          str(aggressiveness),
                          fname,
                          'false'], text=True, stdout=subprocess.PIPE, check=True)
    q = out.stdout.split('\n')[0]
    while re.search('\(\d\.\d+\)', q):
        q = re.sub('\(\d\.\d+\)', '', q)
    q = q.replace('+', '')
    q = q.replace('-', '')
    print(fname)
    print(q)
    return np.array(list(q)).astype(np.int)


def calculate_vad(prediction_path, target_path, aggressiveness=3):
    """
    input: target and prediction audio files;
    output: pandas DF with VAD values
    """

    # run vad per file
    vad_predicted = run_vad(prediction_path, aggressiveness)
    vad_targets = run_vad(target_path, aggressiveness)

    # calculate vad match between target and prediction wav
    vad_score = np.mean(np.equal(vad_predicted, vad_targets))
    return pd.DataFrame({'vad_target': str(vad_targets).replace(' ', ''),
                         'vad_predicted': str(vad_predicted).replace(' ', ''),
                         'vad_match':[vad_score]})