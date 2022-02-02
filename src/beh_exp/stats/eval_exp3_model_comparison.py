'''
Run permutation test on behavioral data from exp 1 and 2

python beh_exp/stats/eval_exp3_model_comparison.py \
    --data_dir /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/subjects/exp3_29subjs \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp3_29subjs \
    --n_perms 1000
'''

import sys
sys.path.insert(0, '.')
import argparse
import os.path as op
import pandas as pd
import seaborn as sns
import glob
import random

from os import makedirs
from matplotlib import pyplot as plt
from beh_exp.process.process_exp1_word_id import parse_info

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def plot_dot_acc(data, title, plot_box=False, plotdir=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    if plot_box:
        sns.boxplot(x='sub', y='weight', data=data, hue='mod', boxprops=dict(alpha=.7), hue_order=['mlp', 'densenet', 'seq2seq'],
                    ax=ax, orient='v', showfliers=False, whis=[5, 95])
    sns.stripplot(x='sub', y='weight', data=data,
                  hue='mod', size=10, orient='v', jitter=0, dodge=True, alpha=.8, ax=ax, hue_order=['mlp', 'densenet', 'seq2seq'])

    plt.title(title)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if plotdir is not None:
        plt.savefig(op.join(plotdir, title + '.pdf'), dpi=160, transparent=True)
        plt.close()


def process_one(data_file, type_exp):
    data_ = pd.read_csv(data_file) # dtype={"user_id": int, "username": "string"}
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
        out['answer'].append(out1[task][i] if data['Response'].values[i] == 'Optie 1' else out2[task][i]) # if option 1 take from out1 otherwise from out2
        out['alternative'].append(out2[task][i] if data['Response'].values[i] == 'Optie 1' else out1[task][i])
        out[task][i] = out['answer'][i]

    # basically choose randomly from 2 options: given respone or alternative
    out['answer'], out['alternative'] = zip(*[random.sample(sublist, 2) for sublist in zip(out['answer'], out['alternative'])])
    out['file'] = [str(int(i)) for i in data['Participant Private ID'].values]
    assert len(set([len(out[k])  for k in out1.keys()])) == 1, 'Unequal number of elements in out dict'
    out_df = pd.DataFrame(out)
    #out_df['file'] = op.basename(data_file).split('.')[0]

    out_df = out_df.astype({"answer": object, "alternative": object})

    return out_df

def run_main(args):

    res_all_files = pd.DataFrame()
    res_perm = pd.DataFrame()
    #out_df_perm = pd.DataFrame() # ends up being huge 6000 rows per perm for 25 subjects

    for p in range(args.n_perms):
        print(p)
        out_df = pd.DataFrame()
        res = pd.DataFrame()

        for f in glob.glob(op.join(args.data_dir, '*.csv')):
            temp = process_one(f, args.type_exp)
            out_df = out_df.append(temp)
            #out_df= out_df.append(process_one(f, args.type_exp))

        out_opt = out_df[out_df['task'] == 'opt']
        out_opt = out_opt.reset_index(drop=True)
        out_opt['weight'] = -0.1
        out_opt.loc[(out_opt.answer == True).values, 'weight'] = 0.1

        for s in [str(i) for i in range(1, 6)]:
            for m in ['mlp', 'densenet', 'seq2seq']:
                for d in out_opt['file'].unique():
                    temp = out_opt[(out_opt['sub']==s) & (out_opt['mod']==m) & (out_opt['file']==d)]
                    res = res.append(pd.DataFrame({'sub': [s], 'mod':[m],
                                                   'weight':[temp['weight'].mean()],
                                                   'file': [d]}))

                temp = res[(res['sub'] == s) & (res['mod'] ==m)]
                res_perm = res_perm.append(pd.DataFrame({'sub': [s], 'mod': [m],
                                               'weight': [temp['weight'].mean()],
                                               'perm': [p]}))

        res['perm'] = p
        res_all_files = res_all_files.append(res)

    res_perm = res_perm.sort_values(by=['sub'])
    res_all_files = res_all_files.sort_values(by=['sub'])

    plotdir = args.data_dir.replace('data', 'pics')
    if not op.isdir(plotdir): makedirs(plotdir)

    plot_dot_acc(res_perm, title='all_dots_perm', plot_box=True, plotdir=plotdir)

    if not op.isdir(args.save_dir): makedirs(args.save_dir)
    res_all_files.to_csv(op.join(args.save_dir, 'results_avg_over_words_opt_perm.csv'))
    res_perm.to_csv(op.join(args.save_dir, 'results_avg_over_words_avg_over_subjs_opt_perm.csv'))


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process exp 3')
    parser.add_argument('--data_dir',  type=str, help='Path to the dir with spreadsheets for exp 3')
    parser.add_argument('--save_dir', type=str, help='Path to results')
    parser.add_argument('--n_perms', '-n', type=int, help='Number of permutations', default=1000)

    args = parser.parse_args()
    args.type_exp = 'exp3'
    run_main(args)
