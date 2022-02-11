'''
Process behvaioral data: experiment 3 (model comparison)

Compmare optimized/non-optimized and forced choice compare reconstructions from different models

Read in directory with all csv spreadsheets (one per subject)
Process each, populate final csv with results
Save csv with results to plot late in Figure 6cd

python beh_exp/process/process_exp3_model_comparison.py \
    --data_dir /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/subjects/exp3_29subjs\
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp3_29subjs
'''
import sys
sys.path.insert(0, '.')
import argparse
import os.path as op
import pandas as pd
import seaborn as sns
import glob
import itertools
from os import makedirs
from matplotlib import pyplot as plt
from beh_exp.process.process_exp1_word_id import parse_info

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def plot_dot_acc(data, title, plot_box=False, plotdir=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    if plot_box:
        sns.boxplot(x='sub', y='weight', data=data, hue='mod', boxprops=dict(alpha=.7),
                    ax=ax, orient='v', showfliers=False, whis=[5, 95], hue_order=['mlp', 'densenet', 'seq2seq'])
    sns.stripplot(x='sub', y='weight', data=data,
                  hue='mod', size=10, orient='v', jitter=0, dodge=True, alpha=.8, ax=ax, hue_order=['mlp', 'densenet', 'seq2seq'])

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if plotdir is not None:
        plt.savefig(op.join(plotdir, title + '.pdf'), dpi=160, transparent=True)
        plt.close()


def process_one(data_file):
    data_ = pd.read_csv(data_file)
    data = data_[['Participant Private ID', 'Zone Type', 'Response',
                  'reconstructions_3.1', 'reconstructions_3.2']]
    data = data[data['Zone Type'] == 'response_button_text']

    # remove rows with trials containing target
    data = data[~data.apply(lambda r: r.str.contains('target').any(), axis=1)]  # test this

    codes = ['sub', 'mod', 'opt', 'word']
    out1 = { k:[] for k in codes}
    out2 = { k:[] for k in codes}
    out = { k:[] for k in ['sub', 'mod', 'opt', 'word'] + ['task', 'answer', 'alternative']}
    for code in codes:
        out1[code] = parse_info(data['reconstructions_3.1'].to_list(), code) # remove empty strings
        out2[code] = parse_info(data['reconstructions_3.2'].to_list(), code) # remove empty strings
        out[code] = parse_info(data['reconstructions_3.1'].to_list(), code) # remove empty strings

    for i in range(len(out1['sub'])):
        if out1['opt'][i] != out2['opt'][i]:
            task = 'opt'
        elif out1['mod'][i] != out2['mod'][i]:
            task = 'mod'
        else:
            raise ValueError
        out['task'].append(task)
        out['answer'].append(out1[task][i] if data['Response'].values[i] == 'Optie 1' else out2[task][i])
        out['alternative'].append(out2[task][i] if data['Response'].values[i] == 'Optie 1' else out1[task][i])
        out[task][i] = out['answer'][i]

    out['file'] = [str(int(i)) for i in data['Participant Private ID'].values]
    assert len(set([len(out[k])  for k in out1.keys()])) == 1, 'Unequal number of elements in out dict'
    out_df = pd.DataFrame(out)
    #out_df['file'] = op.basename(data_file).split('.')[0]
    return out_df

def run_main(args):
    out_df = pd.DataFrame()

    for f in glob.glob(op.join(args.data_dir, '*.csv')):
        out_df = out_df.append(process_one(f))

    out_opt = out_df[out_df['task']=='opt']
    out_opt = out_opt.reset_index(drop=True)
    out_opt['weight'] = -0.1
    out_opt.loc[(out_opt.answer==True).values, 'weight'] = 0.1

    res_opt = pd.DataFrame()
    for s in [str(i) for i in range(1, 6)]:
        for m in ['mlp', 'densenet', 'seq2seq']:
            for d in out_df['file'].unique():
                temp = out_opt[(out_opt['sub']==s) & (out_opt['mod']==m) & (out_opt['file']==d)]
                res_opt = res_opt.append(pd.DataFrame({'sub': [s], 'mod':[m], 'file': [d],
                                               'weight':[temp['weight'].mean()]}))
    res2_opt = pd.DataFrame()
    for s in [str(i) for i in range(1, 6)]:
        for m in ['mlp', 'densenet', 'seq2seq']:
            for w in out_df['word'].unique():
                temp = out_opt[(out_opt['sub']==s) & (out_opt['mod']==m) & (out_opt['word']==w)]
                res2_opt = res2_opt.append(pd.DataFrame({'sub': [s], 'mod':[m], 'word': [w],
                                               'weight':[temp['weight'].mean()]}))


    res_opt = res_opt.sort_values(by=['sub'])
    res2_opt = res2_opt.sort_values(by=['sub', 'mod'])

    plotdir = args.data_dir.replace('data', 'pics')
    if not op.isdir(plotdir): makedirs(plotdir)

    for s in res_opt['file'].unique():
        plot_dot_acc(res_opt[res_opt['file']==s], title=op.basename(s).split('.')[0] + '_opt', plotdir=plotdir)

    plot_dot_acc(res_opt, title='all_dots_opt', plot_box=True, plotdir=plotdir)

    if not op.isdir(args.save_dir): makedirs(args.save_dir)
    out_df.to_csv(op.join(args.save_dir, 'results_all.csv'))
    out_opt.to_csv(op.join(args.save_dir, 'results_opt.csv'))

    res_opt.to_csv(op.join(args.save_dir, 'results_avg_over_words_opt.csv'))
    res2_opt.to_csv(op.join(args.save_dir, 'results_avg_over_subjs.csv'))



##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process exp 1 or 2')
    parser.add_argument('--data_dir',  type=str, help='Path to the dir with spreadsheets for exp 1 or 2')
    parser.add_argument('--save_dir', type=str, help='Path to results')
    args = parser.parse_args()
    run_main(args)
