
import numpy as np
import os
import warnings
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from scipy.signal import resample_poly
import re
import pandas as pd


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    import h5py
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                warnings.warn("Dataset in hdf5 file already exists. "
                                "recreate dataset in hdf5.")
                hdf5_file.__delitem__(hdf5_path)
            else:
                raise Exception("Dataset in hdf5 file already exists. "
                              "if you want to overwrite, please set is_overwrite = True.")
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


def resample(x, sr1=25, sr2=125, axis=0):
    # only real numbers here
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis).astype(np.float32)


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )


def make_save_dir(savedir):
    if os.path.isdir(savedir):
        counter = 1
        savedir0 = savedir
        while os.path.isdir(savedir):
            savedir = savedir0 + str(counter)
            counter += 1
        os.makedirs(savedir)
    else:
        os.makedirs(savedir)
    return savedir


def load_csv_data(f):
    d = pd.read_csv(f)
    data = d.values
    labels = d.columns
    return data, labels


def sec2ind(s, sr):
    return int(Decimal(s * sr).quantize(0, ROUND_HALF_UP))


def flatten_list(t):
    return [item for sublist in t for item in sublist]

def normalize(x, x_mean, x_std):
    return (x - x_mean[None,None,:]) / x_std[None,None,:]


def get_stat_pval(val, baseline):
    return np.sum(val <= baseline) / len(baseline)