'''
for the intense poster
plot box plots for all objective metrics

python stats/plot_objective_results.py \
    --task jip_janneke \
    --n_folds 12 \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/decoding/optuna
'''

import pandas as pd
import json
import argparse
from matplotlib import pyplot as plt
import seaborn as sns
import os.path as op
from utils.private.datasets import get_subjects_by_code

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(args):
    for metric in args.metric:
        results = pd.DataFrame()
        for subject in args.subject:
            for model in args.model:
                gen_dir = op.join(args.res_dir, args.task, subject, model)
                best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))
                trial_dir = op.join(gen_dir,str(best_trial['_number']), 'eval')

                a = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + metric + '.csv'), index_col=[0])
                if metric == 'classify':
                    b = {'target_audio':[], 'reconstructed':[], 'brain_input':[]}
                    for input in b.keys():
                        b[input] = a[a['input'] == input]['accuracy'].mean()
                    b['subject'] = [subject]
                    b['model'] = [model]
                    results = results.append(pd.DataFrame(b))
                else:
                    a['subject'] = subject
                    a['model'] = model
                    results = results.append(a)

        # plot boxplot group per subject and model
        sns.set_context('talk')
        if metric == 'pearsonr':
            sns.boxplot(x='r', y='model', data=results, hue='subject', orient='h')
        elif metric == 'vad':
            sns.boxplot(x='vad_match', y='model', data=results, hue='subject', orient='h')
        elif metric == 'stoi':
            sns.boxplot(x='stoi', y='model', data=results, hue='subject', orient='h')

        elif metric == 'classify':
            sns.barplot(x='reconstructed', y='model', data=results, hue='subject', orient='h')

        if args.plot_dir != '':
            plt.savefig(op.join(args.plot_dir, args.task, 'eval_' + metric + '.pdf'), dpi=160, transparent=True)
            plt.close()



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
                        help='Subject to run')
    parser.add_argument('--metric', '-x', type=str,  nargs="+",
                        choices=['pearsonr', 'vad', 'stoi', 'classify'],
                        default=['pearsonr', 'vad', 'stoi', 'classify'],
                        help='Metric to use')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--res_dir', '-r', type=str, help='Results directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')

    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)
