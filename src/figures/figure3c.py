'''
plot histograms for all optimal parameters

python figures/figure3c.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures
'''

import pandas as pd
import json
import argparse
import seaborn as sns
import os.path as op
import matplotlib.patches as mpatches
import numpy as np

from utils.private.datasets import get_subjects_by_code
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(args):

    params = pd.DataFrame()
    loss = pd.DataFrame()
    for subject in args.subject:
        for model in args.model:
            gen_dir = op.join(args.res_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))
            best_params = best_trial['_params']
            min_loss = {'loss':best_trial['_values']}
            best_params['subject'] = subject
            best_params['model'] = model
            min_loss['subject'] = subject
            min_loss['model'] = model
            params = params.append(pd.DataFrame([best_params]))

##
    #sns.displot(data=params, x='subject', y='model', hue='sr')
    param_names = params.columns.tolist()
    param_names.remove('subject')
    param_names.remove('model')

    for p in param_names:
        print(p)

        fig, ax = plt.subplots(1, 1, figsize=(8,5))

        if p in ['output_type', 'input_band', 'input_ref', 'dense_bottleneck', 'seq_bidirectional']:
            temp = pd.factorize(params[p])[0].reshape(len(args.subject), len(args.model)).T
        else:
            temp = params[p].values.reshape(len(args.subject), len(args.model)).T

        if p in ['sr', 'output_num', 'output_type']:
            c = 'Reds'
        elif p in ['input_band', 'input_ref', 'fragment_len']:
            c = 'Purples'
        elif p in ['learning_rate', 'drop_ratio']:
            c = 'Greys'
        elif 'mlp_' in p:
            c = 'Blues'
        elif 'dense_' in p:
            c = 'Oranges'
        elif 'seq_' in p:
            c = 'Greens'
        else:
            c = 'viridis'

        im = ax.imshow(temp, aspect='auto', cmap=c)
        ax.set_xticks(range(len(args.subject)))
        ax.set_xticklabels(['s1', 's2', 's3', 's4', 's5'], fontsize=28)
        ax.set_yticks(range(len(args.model)))
        ax.set_yticklabels(args.model, fontsize=28)
        ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')
        values = np.unique(temp.ravel())
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i])) for i in range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=28)

        if args.plot_dir != '':
            plt.savefig(op.join(args.plot_dir, 'fig3c_hist_params_' + p + '.pdf'), dpi=160, transparent=True)
            plt.close()

    # sns.displot(params, x='subject', y='model', hue='sr')
    # sns.displot(params, x='subject', y='model', hue='output_type', color=['red', 'blue', 'green'])
    # sns.displot(params, x='subject', y='model', hue='output_num', color=['red', 'blue', 'green'])
    #
    # sns.displot(params, x='subject', y='model', hue='input_ref', palette='Set2')
    # sns.displot(params, x='subject', y='model', hue='input_band', color=['red', 'blue', 'green'])
    # sns.displot(params, x='subject', y='model', hue='fragment_len', color=['red', 'blue', 'green'])
    #
    # sns.displot(params, x='subject', y='model', hue='learning_rate', palette='Reds')
    # sns.displot(params, x='subject', y='model', hue='drop_ratio', palette='husl')
    #
    # sns.displot(params, x='subject', y='model', hue='mlp_n_blocks', palette='Reds', alpha=.5)
    # sns.displot(params, x='subject', y='model', hue='mlp_n_hidden', palette='Reds', alpha=.5)
    #
    # sns.displot(params, x='subject', y='model', hue='dense_bottleneck', palette='Set2')
    # sns.displot(params, x='subject', y='model', hue='dense_growth_rate', palette='Reds')
    # sns.displot(params[params['model']=='densenet'], x='subject', y='model', hue='dense_n_layers', palette='Reds')
    # sns.displot(params, x='subject', y='model', hue='dense_reduce', palette='Reds', alpha=.5)
    #
    # sns.displot(params, x='subject', y='model', hue='seq_bidirectional', palette='bright', alpha=.5)
    # sns.displot(params, x='subject', y='model', hue='seq_n_dec_layers', palette='bright', alpha=.5)
    # sns.displot(params, x='subject', y='model', hue='seq_n_enc_layers', palette='bright', alpha=.5)

##

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
    parser.add_argument('--res_dir', '-r', type=str, help='Results directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')

    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)

##

