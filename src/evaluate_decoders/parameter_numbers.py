'''
Get number of parameters for best models

python evaluate_decoders/parameter_numbers.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna

'''

import sys
sys.path.insert(0, '.')

import os
import os.path as op
import argparse
import torch
import pandas as pd
import json
from utils.private.datasets import get_subjects_by_code


def main(args):
    n_params = pd.DataFrame()
    for isubj, subject in enumerate(args.subject):
        for imod, model in enumerate(args.model):
            gen_dir = op.join(args.res_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
            trial_dir = op.join(gen_dir, str(best_trial))
            model_path = op.join(trial_dir,
                                      op.basename([f.path for f in os.scandir(op.join(gen_dir, str(best_trial))) if
                                                   f.is_dir() and 'eval' not in f.path][0]))
            net = torch.load(op.join(model_path, 'min_val_loss.pth'))
            a = {}
            a['model'] = [model]
            a['subject'] = [subject]
            a['n_params'] = [sum([v.nelement() for k, v in net['model_state_dict'].items() if 'weight' in k])]
            n_params = n_params.append(pd.DataFrame(a))

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
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')

    parser.set_defaults(use_cuda=True)
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)