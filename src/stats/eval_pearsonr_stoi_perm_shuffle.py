'''
Run correlations for permutations of 12 folds in test

Originally saved predictions and targets (.npy) after denormalization: scaling and re-centering output of the model pass,
same was applied to the targets. That operation alone produces very high correlations, and results in a high baseline for
permuted (shifted) data (eval_pearsonr_vad_stoi_perm_shifts).

Calculate correlation without using denormalization, just normalized targets and output from the model


Produces distributions very similar to actual correlations because
    - this is based on 1-second fragments (for word classification) with a long silence bit at the end of almost every trial
    - we mostly pick up on speech vs non-speech rather than detailed word profiles
    plus
    - permutations of 12 usually end up with several items in their right matching place

python stats/eval_pearsonr_stoi_perm_shuffle.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --n_perms 1000

'''

import sys
sys.path.insert(0, '.')

import os
import os.path as op
import argparse
import pandas as pd
import numpy as np
import json

from utils.general import normalize
from utils.plots import plot_eval_metric_box
from utils.private.datasets import get_subjects_by_code
from evaluate_decoders.eval_stoi import calculate_stoi
from evaluate_decoders.eval_pearsonr import calculate_pearsonr_flattened

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def load_t_moments(params):
        return np.load(params['output_mean']), np.load(params['output_std'])


def get_eval_input_params(model_path):
    if op.exists(op.join(model_path, 'eval_input_parameters.txt')):
        params = {}
        with open(op.join(model_path, 'eval_input_parameters.txt'), 'r') as f:
            content = f.read().splitlines()
            for line in content:
                temp = line.split('=')
                params[temp[0]] = temp[1]
        return params
    else:
        raise ValueError

# def normalize(x, x_mean, x_std):
#     return (x - x_mean[None,None,:]) / x_std[None,None,:]


def main(args):
    for isubj, subject in enumerate(args.subject):
        for imod, model in enumerate(args.model):
            gen_dir = op.join(args.res_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
            trial_dir = op.join(gen_dir, str(best_trial), 'eval')
            res_pearsonr_perm = pd.DataFrame()
            res_stoi_perm = pd.DataFrame()
            predictions = []
            targets = []
            model_path = op.basename([f.path for f in os.scandir(op.join(gen_dir, str(best_trial)))
                                                                        if f.is_dir() and 'eval' not in f.path][0])

            for fold in range(12):
                fold_dir = op.join(trial_dir, 'fold' + str(fold))
                assert op.exists(fold_dir), 'No fold dir at ' + fold_dir
                pred = np.load(os.path.join(fold_dir, 'val_predictions.npy'))
                tar = np.load(os.path.join(fold_dir, 'val_targets.npy'))
                input_params = get_eval_input_params(fold_dir)
                t_mean, t_std = load_t_moments(input_params) # normalize predictions and targets again, since saved denormalized ones
                predictions.append(normalize(pred, t_mean, t_std))
                targets.append(normalize(tar, t_mean, t_std))

            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)


            for perm in range(args.n_perms):
                # exclude permutations with any matches + check it's unique
                print(perm)
                p = np.random.permutation(12)
                if not res_pearsonr_perm.empty: t = np.array(res_pearsonr_perm['perm'].to_list()).flatten().reshape(-1, 12)
                while np.any(p == np.arange(12)) or \
                                    (not res_pearsonr_perm.empty and list(p) in t.tolist()):
                    p = np.random.permutation(12)

                # pearsonr
                predictions_perm = predictions.copy()[p]
                res = {'subject': subject, 'model': model, 'perm': [p], 'pearsonr': []}
                res['pearsonr'] = calculate_pearsonr_flattened(predictions_perm, targets)['r'].mean()
                res_pearsonr_perm = res_pearsonr_perm.append(pd.DataFrame(res), ignore_index=True)

                # stoi
                res = {'subject': [subject], 'model': [model], 'perm': [p], 'stoi': [], 'estoi': []}
                for ifold, fold in enumerate(p):
                    fold_dir1 = op.join(trial_dir, 'fold' + str(fold))
                    fold_dir2 = op.join(trial_dir, 'fold' + str(ifold))
                    audio_val_targets = os.path.join(fold_dir1, model_path, 'parallel_wavegan', 'targets_validation_gen.wav')
                    audio_val_predictions = os.path.join(fold_dir2, model_path, 'parallel_wavegan', model + '_validation_gen.wav')
                    temp = calculate_stoi(audio_val_predictions, audio_val_targets, pad=True)
                    res['stoi'] = temp['stoi'].mean()
                    res['estoi'] = temp['estoi'].mean()
                res_stoi_perm = res_stoi_perm.append(pd.DataFrame(res), ignore_index=True)


            plotdir = trial_dir.replace('results', 'pics/decoding')
            for metric in args.metric:
                if metric == 'pearsonr':
                    res = res_pearsonr_perm
                elif metric == 'stoi':
                    res = res_stoi_perm
                else:
                    raise ValueError
                res.to_csv(os.path.join(trial_dir, 'eval_n_folds12_' + metric + '_perm_shuf_words.csv'))
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
                        choices=['pearsonr', 'stoi'],
                        default=['pearsonr', 'stoi'],
                        help='Metric to use')
    parser.add_argument('--n_perms', '-n', type=int, help='Number of permutations', default=1000)
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')

    parser.set_defaults(use_cuda=True)
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)
