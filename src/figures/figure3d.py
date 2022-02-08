'''
plot parameter importances

python figures/figure3d.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures
'''

import pandas as pd
import argparse
import os.path as op
import optuna

from utils.private.datasets import get_subjects_by_code
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(args):

    for model in args.model:
        res = pd.DataFrame()
        for subject in args.subject:
            study_name = args.task + '_' + subject + '_' + model
            res_dir = op.join(args.res_dir, args.task, subject, model)
            study = optuna.load_study(study_name=study_name,
                                      storage='sqlite:///' + op.join(res_dir, study_name + '.db'))

            importances = optuna.importance.get_param_importances(study)
            importances['subject'] = subject
            importances['model'] = model
            res = res.append(pd.DataFrame(importances, index=[0]))

        r = res[res['model']==model].drop(columns=['model'])
        first_cols = ['input_ref', 'input_band', 'fragment_len', 'output_num', 'output_type', 'sr', 'drop_ratio', 'learning_rate']
        r = r[first_cols + list(set(r.columns.to_list()) - set(first_cols))]
        #c = sns.color_palette("Set1")
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        r.set_index('subject').T.plot(kind='bar', stacked=True, edgecolor='black', linewidth=0.5, ax=ax)
        plt.ylabel('Importance')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.legend([], [], frameon=False)

        if args.plot_dir != '':
            plt.savefig(op.join(args.plot_dir, 'fig3d_importances_' + model + '.pdf'),
                        dpi=160,
                        transparent=True)
            plt.close()

##

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameter importance')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Subject to run')
    parser.add_argument('--res_dir', '-r', type=str, help='Results directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')

    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)