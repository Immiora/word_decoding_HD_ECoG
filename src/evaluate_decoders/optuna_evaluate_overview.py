'''
Added other classifier options. This needs to be tested if rerunning test + eval

Put together evaluation results across folds into one plot and one output file

python evaluate_decoders/optuna_evaluate_overview.py \
    --eval_path path_to_optuna_$task_$subject_$model_$trial_eval
    --n_folds 12
'''


import sys
sys.path.insert(0, '.')

import os
import argparse
import pandas as pd

from utils.plots import plot_eval_metric_box

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(args):

    # set up output
    if args.save_dir == '':
        args.save_dir = args.eval_path

    # set up plotting
    plotdir = args.save_dir.replace('results', 'pics/decoding')
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    if not os.path.isdir(plotdir): os.makedirs(plotdir)

    for metric in args.metric:
        fold_path = os.path.join(args.eval_path, 'fold' + str(0))
        if metric == 'classify':
            metric = metric + '_' + args.clf_type
        metric_path = os.path.join(fold_path, 'eval_' + metric + '.csv')
        res = pd.read_csv(metric_path, index_col=0).assign(fold=0)

        for k_fold in range(1, args.n_folds):
            res = res.append(pd.read_csv(metric_path.replace('fold0', 'fold' + str(k_fold)), index_col=0).assign(fold=k_fold))

        if metric == 'pearsonr':
            res = res.reset_index()
            plot_eval_metric_box(metric, res[['mel','r']], plotdir, by='mel')
            print('pearsonr median over mel: ', res.groupby(res.mel)[['r']].median())

        elif metric == 'vad':
            plot_eval_metric_box(metric, res[['vad_match']], plotdir)

        elif metric == 'stoi':
            plot_eval_metric_box(metric, res[['stoi', 'estoi']], plotdir)

        elif metric == 'classify':
            plot_eval_metric_box(metric, res[['input','accuracy']], plotdir, by='input')
            print('median accuracy over inputs: ', res.groupby(res.input)[['accuracy']].median())

        res.to_csv(os.path.join(args.save_dir, 'eval_n_folds' + str(args.n_folds) + '_' + metric + '.csv'))
        print('Done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--eval_path', '-p', type=str, help='Directory with eval folder')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--metric', '-m', type=str,  nargs="+",
                        choices=['pearsonr', 'vad', 'stoi', 'classify'],
                        default=['pearsonr', 'vad', 'stoi', 'classify'],
                        help='Metric to use')
    parser.add_argument('--clf_type', '-c', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='svm_linear',
                        help='Type of classifier')
    parser.add_argument('--save_dir', '-s', type=str, help='Output directory', default='')

    args = parser.parse_args()

    main(args)