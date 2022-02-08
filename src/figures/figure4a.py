'''
Plot spectrograms


python figures/figure4a.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --n_folds 12
'''

import sys
sys.path.insert(0, '.')

import os
import os.path as op
import argparse
import numpy as np
import json
#import soundfile as sf
import librosa
import librosa.display

from utils.private.datasets import get_subjects_by_code
from matplotlib import pyplot as plt
#from prepare_data.options_grid_spectrogram_jip_janneke import logmelfilterbank


def load_audio(fname):
    x, sr = librosa.load(fname, sr=22050)
    return x

def wav2spec(x):
    X = librosa.stft(x, win_length=1024)
    Xdb = librosa.amplitude_to_db(abs(X))
    return Xdb

def main(args):

    for isubj, subject in enumerate(args.subject):
        print(subject)

        for imod, model in enumerate(args.model):
            gen_dir = op.join(args.res_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
            trial_dir = op.join(gen_dir, str(best_trial), 'eval')
            model_dir = op.basename(
                [f.path for f in os.scandir(op.join(gen_dir, str(best_trial))) if f.is_dir() and 'eval' not in f.path][0])

            targets_2d, reconstructions_2d = [], []
            targets_wav, reconstructions_wav = [], []
            for fold in range(args.n_folds):
                fold_dir = op.join(trial_dir, 'fold' + str(fold))
                assert op.exists(fold_dir), 'No fold dir at ' + fold_dir
                audio_val_targets = op.join(fold_dir, model_dir, 'parallel_wavegan', 'targets_validation_gen.wav')
                audio_val_predictions = op.join(fold_dir, model_dir, 'parallel_wavegan',
                                                          model + '_validation_gen.wav')
                t = load_audio(audio_val_targets)
                p = load_audio(audio_val_predictions)
                targets_2d.append(wav2spec(t))
                reconstructions_2d.append(wav2spec(p))
                targets_wav.append(librosa.util.normalize(t))
                reconstructions_wav.append(librosa.util.normalize(p))

            plt.figure(figsize=(12, 5))
            plt.subplot(211)
            plt.plot(np.concatenate(targets_wav))
            plt.subplot(212)
            plt.plot(np.concatenate(reconstructions_wav))

            if args.plot_dir != '':
                plt.savefig(op.join(args.plot_dir, 'fig4a_waveform_' + subject + '_' + model + '.pdf'),
                            dpi=160, transparent=True)
                plt.close()

            # plt.figure(figsize=(12, 5))
            # plt.subplot(211)
            # librosa.display.specshow(np.hstack(targets_2d), sr=22050, x_axis='time', y_axis='hz', cmap='rainbow')
            # plt.ylim(0, 7600)
            # plt.subplot(212)
            # librosa.display.specshow(np.hstack(reconstructions_2d), sr=22050, x_axis='time', y_axis='hz', cmap='rainbow')
            # plt.ylim(0, 7600)
            #
            # if args.plot_dir != '':
            #     plt.savefig(op.join(args.plot_dir, 'fig4a_reconstructed_spectrogram_' + subject + '_' + model + '.pdf'),
            #                 dpi=160, transparent=True)
            #     plt.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Model to run')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)
    main(args)
