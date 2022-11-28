'''

Obtain ceiling estimates for each evaluation metric: pearsonr, vad and stoi

Vad, stoi are computed on actual audio (original clean or raw wav) and synthesized targets
Pearsonr is computed by taking synthesize target, extracting its spectrogram, downsampling it

python stats/eval_pearsonr_vad_stoi_audio_ceiling.py \
    --task jip_janneke \
    --metric pearsonr \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna

'''

import sys
sys.path.insert(0, '.')

import os
import os.path as op
import argparse
import pandas as pd
import numpy as np
import json
import tempfile
import librosa
import copy
import soundfile as sf
import torch

from utils.arguments import load_params
from utils.plots import plot_eval_metric_box
from utils.private.datasets import get_subjects_by_code
from evaluate_decoders.eval_vad import calculate_vad
from evaluate_decoders.eval_stoi import calculate_stoi
from evaluate_decoders.eval_pearsonr import calculate_pearsonr_flattened
from prepare_data.options_grid_spectrogram_jip_janneke import logmelfilterbank

from utils.datasets import BrainDataset
from torch.utils.data import Dataset, DataLoader

def get_audio_files(model_path):
    if op.exists(op.join(model_path, 'eval_input_parameters.txt')):
        params = {}
        with open(op.join(model_path, 'eval_input_parameters.txt'), 'r') as f:
            content = f.read().splitlines()
            for line in content:
                temp = line.split('=')
                params[temp[0]] = temp[1]
        type_audio = 'clean' if 'clean' in params['output_file'] else 'raw'
        json_audio_file = op.join('/Fridge/users/julia/project_decoding_jip_janneke/data',
                             params['subject'], params['subject'] + '_audio_' + params['task'] + '.json')
        return params['output_file'], json.load(open(json_audio_file, 'r'))[type_audio]

def parse_ouput_file(output_file):
    params = {}
    for k in ['nfft', 'hop', 'mel']:
        params[k] = int(output_file.split('_'+k)[1].split('_')[0])
    return params

def load_original_audio(audio_path):
    a, sr = librosa.load(audio_path, sr=22050)
    return a, sr

def match_wav2_to_wav1(wav2, wav1):
    if wav2.shape[0] < wav1.shape[0]:
        wav2 = np.pad(wav2, [0, wav1.shape[0] - wav2.shape[0]], constant_values=0)
    else:
        wav2 = wav2[:len(wav1)]
    return wav2

def main(args):

    for isubj, subject in enumerate(args.subject):
        for imod, model in enumerate(args.model):
            gen_dir = op.join(args.res_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
            trial_dir = op.join(gen_dir, str(best_trial), 'eval')
            res_pearsonr, res_vad, res_stoi = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            for fold in range(12):
                print(fold)
                fold_dir = op.join(trial_dir, 'fold' + str(fold))
                model_path = op.join(fold_dir,
                                          op.basename([f.path for f in os.scandir(op.join(gen_dir, str(best_trial))) if
                                                       f.is_dir() and 'eval' not in f.path][0]))

                audio_val_targets = os.path.join(model_path, 'parallel_wavegan', 'targets_validation_gen.wav')
                output_file, audio_original_file = get_audio_files(fold_dir)
                tmp_perm_dir = tempfile.mkdtemp()
                resynth_audio, sr = librosa.load(audio_val_targets, sr=22050)

                # if 'pearsonr' in args.metric: # should not use this
                #    raise ValueError
                    # from evaluate_decoders.eval_pearsonr import calculate_pearsonr_flattened
                    # from stats.eval_pearsonr_stoi_perm_shuffle import load_t_moments, normalize
                    # from prepare_data.options_grid_spectrogram_jip_janneke import logmelfilterbank
                    # val_targets = np.load(os.path.join(fold_dir, 'val_targets.npy'))
                    # t_mean, t_std = load_t_moments(fold_dir)
                    # val_targets_z = normalize(val_targets, t_mean, t_std)
                    #
                    # resynth_audio = np.pad(resynth_audio, (0, sr-len(resynth_audio)))
                    # mel_params = parse_ouput_file(output_file)
                    # resynth_mel = logmelfilterbank(resynth_audio, sr, fft_size=mel_params['nfft'],
                    #                                hop_size=mel_params['hop'],
                    #                                fmin=80,
                    #                                fmax=7600,
                    #                                num_mels=mel_params['mel'])
                    # val_resynth = resynth_mel[None,:val_targets_z.shape[1]]
                    # val_resynth_z = normalize(val_resynth, t_mean, t_std)
                    #
                    # # pearson correlation
                    # print('Computing pearsonr')
                    # temp = calculate_pearsonr_flattened(val_targets, val_resynth) # does not really make sense to do this
                    # res_pearsonr = res_pearsonr.append(temp, ignore_index=True)

                #if 'vad' in args.metric or 'stoi' in args.metric:
                fragments_file = op.join(fold_dir, 'fragments_fold'+str(fold)+'_speech_only.csv')

                audio_val_original = os.path.join(tmp_perm_dir, 'temp_audio.wav')
                audio_original, sr_ori = load_original_audio(audio_original_file)

                hargs = copy.copy(args)
                hargs.model_path = model_path
                hargs = load_params(hargs)
                dataset = BrainDataset(hargs.input_file, hargs.output_file, fragments_file,
                                       hargs.input_sr, hargs.output_sr, hargs.fragment_len, hargs.input_delay)
                dataset.audio_data = np.arange(dataset.audio_data.shape[0])[:, None]
                val_dataset = torch.utils.data.Subset(dataset, dataset.fragments.index[
                    dataset.fragments['subset'] == 'validation'].tolist())
                val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,  num_workers=0)
                _, temp, _ = next(iter(val_loader))
                temp = temp.detach().cpu().numpy().flatten()
                word_on = int(round(temp[0] * sr_ori / hargs.sr))
                word_off = int(round((temp[-1]+1) * sr_ori / hargs.sr))
                temp_ori = audio_original[word_on:word_off]

                assert sr == sr_ori, 'Sampling rates do not match'
                print('Original audio len: ' + str(temp_ori.shape[0]))
                print('Synthesized audio len: ' + str(resynth_audio.shape[0]))
                temp_ori2 = match_wav2_to_wav1(temp_ori, resynth_audio)
                sf.write(audio_val_original, temp_ori2, sr)

                assert os.path.exists(audio_val_targets), 'File does not exist: ' + audio_val_targets
                assert os.path.exists(audio_val_original), 'File does not exist: ' + audio_val_original

                # Pearson R
                if 'pearsonr' in args.metric:
                    x, a = sf.read(audio_val_targets)
                    y, a = sf.read(audio_val_original)
                    target_mel = logmelfilterbank(x, sr, fft_size=525, # not really correct to ignore original params, but best possible
                                                   hop_size=294,
                                                   fmin=80,
                                                   fmax=7600,
                                                   num_mels=40)
                    original_mel = logmelfilterbank(y, sr, fft_size=525,
                                                   hop_size=294,
                                                   fmin=80,
                                                   fmax=7600,
                                                   num_mels=40)
                    temp = calculate_pearsonr_flattened(target_mel[None,:,:],
                                                        original_mel[None,:,:])
                    res_pearsonr = res_pearsonr.append(temp, ignore_index=True)

                # VAD
                if 'vad' in args.metric:
                    print('Computing VAD match')
                    temp = calculate_vad(audio_val_targets, audio_val_original)
                    res_vad = res_vad.append(temp, ignore_index=True)

                # STOI
                if 'stoi' in args.metric:
                    print('Computing STOI/ESTOI')
                    temp = calculate_stoi(audio_val_targets, audio_val_original)
                    res_stoi = res_stoi.append(temp, ignore_index=True)

            plotdir = trial_dir.replace('results', 'pics/decoding')
            for metric in args.metric:
                if metric == 'pearsonr':
                    res = res_pearsonr
                elif metric == 'vad':
                    res = res_vad
                elif metric == 'stoi':
                    res = res_stoi
                else:
                    raise ValueError
                res.to_csv(os.path.join(trial_dir, 'eval_n_folds12_'+ metric+'_audio_ceil.csv'))
                plot_eval_metric_box(metric, res, plotdir=plotdir)



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
    parser.add_argument('--metric', '-e', type=str,  nargs="+",
                        choices=['pearsonr', 'vad', 'stoi'],
                        default=['pearsonr', 'vad', 'stoi'],
                        help='Metric to use')
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)
