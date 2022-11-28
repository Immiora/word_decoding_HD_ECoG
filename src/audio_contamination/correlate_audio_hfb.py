'''
run the audio cotamination analysis: correlation between audio and hfb per frequency bin
using only time preprocessed ecog data: car-referenced, at 2000 Hz
using raw audio data

python audio_contamination/correlate_audio_hfb.py \
    --task jip_janneke \
    --subject_code xoop \
    --ref car \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/audio_contamination
'''

import sys
sys.path.insert(0, '.')


import json
import librosa
import librosa.display
import pandas as pd
import argparse
import os.path as op
import numpy as np
from os import makedirs
from utils.general import get_stat_pval
from scipy.stats import pearsonr
from utils.general import sec2ind, load_csv_data
from utils.private.datasets import get_subjects_by_code

def main(args):

    if not op.isdir(args.save_dir): makedirs(args.save_dir)
    data_dir = op.join('/Fridge/users/julia/project_decoding_' + args.task, 'data')
    sr_input = 2000
    sr_output = 16000
    n_freq_bins = 80 # not Hz, bins: np.linspace(0, (sr_input + 1) / 2, 257)[:n_freq_bins] Hz are used

    for subject in args.subject:
        print(subject)

        for ref in args.ref:
            # load brain data
            input_file = op.join(data_dir, subject, 'time_' + args.task + '_ecog_' + ref + '_nofreq_noz_2000Hz.csv')
            input_data, input_labels = load_csv_data(input_file)
            print('input data: ', input_data.shape[0]/sr_input, 's')

            # load audio data
            output_info = json.loads(open(op.join(data_dir, subject, subject + '_audio_' + args.task + '.json'), 'r').read())
            output_file = output_info['raw']
            output_data, sr_raw = librosa.load(output_file, sr=None)
            output_data_res = librosa.resample(output_data, sr_raw, sr_output)
            print('output data: ', output_data_res.shape[0]/sr_output, 's')

            # load word onsets and offsets
            word_info = json.loads(open(op.join(data_dir, subject,
                                subject + '_contexts_' + args.task + '_15Hz_step0.0667s_window0.1334s.json'), 'r').read())
            word_file = word_info['subsets_path']
            word_fragments = pd.read_csv(word_file, header=0)

            # get audio sfft
            A = librosa.amplitude_to_db(np.abs(librosa.stft(output_data_res, n_fft=4096)), ref=np.max)[:n_freq_bins]
            print('sfft data: ', A.shape[-1]/(sr_output/(4096/4)), 's') # hop size is 4
            rs = np.zeros((input_data.shape[1], len(word_fragments), n_freq_bins)) # channels x words x analyzed frequencies
            ps = np.zeros((input_data.shape[1], len(word_fragments), n_freq_bins))
            rs_perm = np.zeros((args.n_perm * len(word_fragments), input_data.shape[1], n_freq_bins)) # nperm x channels x words x analyzed frequencies
            ps_perm = np.zeros((args.n_perm * len(word_fragments), input_data.shape[1], n_freq_bins))

            # calculate correlation: default was [word_onset - .5 s ; word_onset + 1 s]
            # but speech on/off leads to additional false correlation?
            start_offset = 0 #-.5
            end_offset = 0 #1
            for ch in range(input_data.shape[1]):
                print(ch)
                E = librosa.amplitude_to_db(np.abs(librosa.stft(input_data[:, ch], n_fft=512)), ref=np.max)[:n_freq_bins]

                for iword, row in word_fragments.iterrows():
                    word_onset = max(sec2ind(row['xmin'] + start_offset, sr_output/(4096/4)), 0) # new sr is sr_output/(4096/4)
                    word_offset = min(sec2ind(row['xmax'] + end_offset, sr_output/(4096/4)), E.shape[-1])
                    temp = np.array([pearsonr(ia, ie) for ia, ie in zip(A[:, word_onset:word_offset],
                                                                         E[:, word_onset:word_offset])]).T
                    rs[ch, iword] = temp[0]
                    ps[ch, iword] = temp[1]

                    # for iperm in range(args.n_perm):
                    #     time_shift = np.random.randint(75, E.shape[1] - 75) * np.random.choice([-1, 1], 1)[0]
                    #     temp = np.array([pearsonr(ia, ie) for ia, ie in zip(A[:, word_onset:word_offset],
                    #                                         np.roll(E, time_shift, 1)[:, word_onset:word_offset])]).T
                    #     rs_perm[iperm, ch, iword] = temp[0]
                    #     ps_perm[iperm, ch, iword] = temp[1]
                for iperm in range(args.n_perm * len(word_fragments)):
                    time_shift = np.random.randint(75, E.shape[1]-75) * np.random.choice([-1, 1], 1)[0]
                    row = word_fragments.sample(1).iloc[0]
                    word_onset = max(sec2ind(row['xmin'] + start_offset, sr_output/(4096/4)), 0) # new sr is sr_output/(4096/4)
                    word_offset = min(sec2ind(row['xmax'] + end_offset, sr_output/(4096/4)), E.shape[-1])
                    temp = np.array([pearsonr(ia, ie) for ia, ie in zip(A[:, word_onset:word_offset],
                                                        np.roll(E, time_shift, 1)[:, word_onset:word_offset])]).T
                    rs_perm[iperm, ch] = temp[0]
                    ps_perm[iperm, ch] = temp[1]

            rs[np.isnan(rs)] = 0
            ps[np.isnan(ps)] = -1
            rs_perm[np.isnan(rs_perm)] = 0
            ps_perm[np.isnan(ps_perm)] = -1

            rs_perm = rs_perm.rehsape((args.n_perm, len(word_fragments), input_data.shape[1], n_freq_bins)).transpose(1, 2)

            # statistical testing
            stat_ps = np.zeros((input_data.shape[1], n_freq_bins))
            for ch in range(rs.shape[0]):
                for f in range(rs.shape[-1]):
                    stat_ps[ch, f] = get_stat_pval(np.mean(rs[ch, :, f]), np.mean(rs_perm[:, ch, :, f], -1))

            #
            # import matplotlib
            # matplotlib.use('TkAgg')
            from matplotlib import pyplot as plt
            plt.imshow(stat_ps, aspect='auto')


            # save result
            np.save(op.join(args.save_dir, subject + '_pearson_cor_' + ref + '.npy'), rs.astype(np.float16))
            np.save(op.join(args.save_dir, subject + '_pearson_pval_' + ref + '.npy'), ps.astype(np.float16))

            np.save(op.join(args.save_dir, subject + '_pearson_cor_' + ref + '_perm.npy'), rs_perm.astype(np.float16))
            np.save(op.join(args.save_dir, subject + '_pearson_stat_pval_' + ref + '_perm.npy'), stat_ps.astype(np.float16))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--ref', '-r', type=str,  nargs="+", choices=['car', 'bipolar', 'noref'])
    parser.add_argument('--n_perm', type=int, default=100)
    parser.add_argument('--save_dir', '-o', type=str, help='Output directory')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)
    main(args)



