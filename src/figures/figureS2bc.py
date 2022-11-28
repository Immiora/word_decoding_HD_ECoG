'''
plot correlation of audio and brain data for the contamination analysis: 1 electrode
data from run_contamination_audio

python figures/figureS2bc.py \
    --task jip_janneke \
    --ref car bipolar noref \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/audio_contamination \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures


'''
import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import argparse
import pandas as pd
from utils.private.datasets import get_subjects_by_code


def main(args):
    sr_input = 2000
    tval_speech_dir = '/Fridge/users/julia/project_decoding_jip_janneke/results/ttest_speech_nonspeech'

    for subject in args.subject:

        for ref in args.ref:
            # load data
            rs = np.load(op.join(args.res_dir, subject + '_pearson_cor_' + ref + '.npy')).astype(np.float32)

            # load t-values for plotting 1e correlation profile
            speech_tvals = pd.read_csv(op.join(tval_speech_dir, subject + '.csv'), header=0)
            el = speech_tvals.sort_values(by=['t-value'], ascending=False).reset_index(drop=True).loc[0, 'electrodes']

            # plot 1 electrode
            plt.figure(figsize=(6, 2), dpi=160)
            plt.plot(rs[el].T, color='grey', linewidth=.1, alpha=.5)
            plt.plot(np.mean(rs[el], 0), color='k', linewidth=2.5)
            plt.xticks(range(rs.shape[-1])[::5],
                       np.round(np.linspace(0, (sr_input + 1) / 2, 257)[:rs.shape[-1]]).astype(np.int)[::5],
                       rotation=45,
                       fontsize=6)
            plt.savefig(op.join(args.plot_dir,
                        'figS2b_audio_contamination_el' + str(el) + '_cor_' + subject + '_' + ref +'_word_onset_offset.pdf'),
                        dpi=160, transparent=True)
            plt.close()


            # plot mean over all electrodes and freq
            plt.figure(figsize=(8, 6), dpi=160)
            plt.imshow(np.mean(rs, 1), aspect='auto', vmin=0, vmax=.8, cmap='Reds')
            plt.xticks(range(rs.shape[-1])[::5], # 257 is 512/2 + 1 = top frequency for sfft for nfft=512
                       np.round(np.linspace(0, (sr_input + 1) / 2, 257)[:rs.shape[-1]]).astype(np.int)[::5])
            plt.colorbar()
            plt.savefig(op.join(args.plot_dir,
                        'figS2c_audio_contamination_all_cormat_' + subject + '_' + ref + '_word_onset_offset.pdf'),
                         dpi=160, transparent=True)
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--ref', '-r', type=str,  nargs="+", choices=['car', 'bipolar', 'noref'])
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)
    main(args)
