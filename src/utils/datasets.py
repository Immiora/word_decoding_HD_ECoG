import numpy as np
import torch
import pandas as pd
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from decimal import Decimal, ROUND_HALF_UP


class BrainDataset(Dataset):
    def __init__(self, input_file, output_file, fragments_file, brain_sr, audio_sr, frag_len, brain_delay):
        self.fragments = pd.read_csv(fragments_file, index_col=False)
        self.brain_sr = brain_sr
        self.audio_sr = audio_sr
        self.frag_len = frag_len
        self.brain_delay = brain_delay

        # brain data
        brain_data = np.load(input_file)
        print('Input data loaded')
        self.brain_data = np.roll(brain_data, -int(round(brain_sr * brain_delay)), axis=0)
        print('Input data shifted by ' + str(brain_delay))

        # mel data
        self.audio_data = np.load(output_file)
        print('Output data loaded')

    def __getitem__(self, index):
        # temp = self.audio_data[audio_onset:audio_onset + int(self.frag_len / (1 / self.audio_sr))]
        # t = temp[int(Decimal(temp.shape[0] / 2).quantize(0, ROUND_HALF_UP))][None, :]  # take one middle point
        # x = self.brain_data[brain_onset:brain_onset + int(self.frag_len / (1 / self.brain_sr))]
        sec2ind = lambda s, sr: int(Decimal(s * sr).quantize(0, ROUND_HALF_UP))
        samples2ind = lambda s: int(Decimal(s).quantize(0, ROUND_HALF_UP))  # take care of rounding up for indices

        audio_onset = sec2ind(self.fragments.iloc[index]['xmin'], self.audio_sr)
        brain_onset = sec2ind(self.fragments.iloc[index]['xmin'], self.brain_sr)
        audio_offset = audio_onset + samples2ind(self.frag_len / (1 / self.audio_sr))
        brain_offset = brain_onset + samples2ind(self.frag_len / (1 / self.brain_sr))

        temp = self.audio_data[audio_onset:audio_offset]
        t = temp[max(samples2ind((temp.shape[0] - 1) / 2), 0)][None,:]  # take one middle point: subtract 1 because of 0-index
        x = self.brain_data[brain_onset:brain_offset]

        return torch.Tensor(x), torch.Tensor(t), index

    def __len__(self):
        return len(self.fragments)


def split_data(dataset):
    trainset = torch.utils.data.Subset(dataset, dataset.fragments.index[dataset.fragments['subset'] == 'train'].tolist())
    valset = torch.utils.data.Subset(dataset, dataset.fragments.index[dataset.fragments['subset'] == 'validation'].tolist())
    testset = torch.utils.data.Subset(dataset, dataset.fragments.index[dataset.fragments['subset'] == 'test'].tolist())
    return trainset, valset, testset


def load_data(trainset, valset, testset, batch_size, shuffle_train=True): # changed August 20, double-check this works!
    drop_last = False
    if len(trainset) % batch_size == 1:
        drop_last = True

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train, num_workers=0, drop_last=drop_last)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def get_input_mean_std(trainset, use_pca=False, n_pcs=None):
    train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
    x_temp, t_temp, idx = next(iter(train_loader))
    x_temp = x_temp.numpy().reshape(-1, x_temp.shape[-1])
    x_scaler = StandardScaler()
    x_scaler.fit(x_temp)
    if use_pca:
        x_scaled = x_scaler.transform(x_temp)
        pca = PCA(random_state=100, n_components=n_pcs)
        pca.fit(x_scaled)
        # scale again after PCA?
        return x_scaler.mean_, x_scaler.scale_, pca
    else:
        return x_scaler.mean_, x_scaler.scale_


def get_output_mean_std(trainset):
    train_loader = DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=0)
    x_temp, t_temp, idx = next(iter(train_loader))
    t_temp = t_temp.numpy().reshape(-1, t_temp.shape[-1])
    t_scaler = StandardScaler()
    t_scaler.fit(t_temp)
    return t_scaler.mean_, t_scaler.scale_


def get_moments(trainset, x_mean_file, x_std_file, t_mean_file, t_std_file, use_pca, n_pcs, can_write=False):
    """
        can_write: False, mean for evaluation, do not calculate and write moments if they do not exist
    """
    if use_pca:
        import pickle
    pca = None

    if not os.path.isfile(t_mean_file) or not os.path.isfile(t_std_file):
        if can_write:
            t_mean, t_std = get_output_mean_std(trainset)
            np.save(t_mean_file, t_mean)
            np.save(t_std_file, t_std)
            print('Output moments calculated')
        else:
            raise Exception('Output moment files do not exist and writing was not enabled')

    if not os.path.isfile(x_mean_file) or not os.path.isfile(x_std_file):
        if can_write:
            if use_pca:
                x_mean, x_std, pca = get_input_mean_std(trainset, use_pca=use_pca, n_pcs=n_pcs)
                pickle.dump(pca, open(x_mean_file.replace('.npy', '_' + str(n_pcs) + 'pc.p'), 'wb'))
            else:
                x_mean, x_std = get_input_mean_std(trainset, use_pca=use_pca, n_pcs=n_pcs)
            np.save(x_mean_file, x_mean)
            np.save(x_std_file, x_std)
            print('Input moments calculated')
        else:
            raise Exception('Input moment files do not exist and writing was not enabled')

    x_mean = torch.Tensor(np.load(x_mean_file))
    x_std = torch.Tensor(np.load(x_std_file))
    t_mean = torch.Tensor(np.load(t_mean_file))
    t_std = torch.Tensor(np.load(t_std_file))
    if use_pca:
        pca = pickle.load(open(x_mean_file.replace('.npy', '_' + str(n_pcs) + 'pc.p'), 'rb'))
    print('Moments loaded')

    return x_mean, x_std, t_mean, t_std, pca