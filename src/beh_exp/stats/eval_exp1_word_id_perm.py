'''
Run permutation test on behavioral data from exp 1 and 2

python beh_exp/stats/eval_exp1_word_id_perm.py \
    --data_dir /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/subjects/exp1_30subjs \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp1_30subjs \
    --n_perms 1000
'''

import argparse
import os.path as op
import pandas as pd
import seaborn as sns
import glob
import random

from os import makedirs
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def plot_dot_acc(data, title, plot_box=False, plotdir=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    if plot_box:
        sns.boxplot(x='sub', y='accuracy', data=data[data['recon'] == True], hue='mod', boxprops=dict(alpha=.7),
                    ax=ax, orient='v', showfliers=False, whis=[5, 95], hue_order=['mlp', 'densenet', 'seq2seq'])
        # sns.boxplot(x='sub', y='accuracy', data=data[data['recon'] == False], hue='mod', boxprops=dict(alpha=.3),
        #             orient='v',
        #             ax=ax, color='gray', showfliers=False, whis=[5, 95])
    sns.stripplot(x='sub', y='accuracy', data=data[data['recon'] == True], hue_order=['mlp', 'densenet', 'seq2seq'],
                  hue='mod', size=10, orient='v', jitter=0, dodge=True, alpha=.8, ax=ax)
    # sns.stripplot(x='sub', y='accuracy', data=data[data['recon'] == False],
    #               hue='mod', size=10, orient='v', jitter=0, dodge=True, color='grey', alpha=0.3, ax=ax)

    plt.title(title)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if plotdir is not None:
        plt.savefig(op.join(plotdir, title + '.pdf'), dpi=160, transparent=True)
        plt.close()


def parse_info(l, code):
    """
    Args:
        l: list of strings from reconstructions_1
        code: sub, mod, opt

    Returns:
        list of info per specified code
    """
    if code == 'recon':
        return [True if'recon' in i else False for i in l]
    elif code == 'opt':
        return [i.split(code + '-')[1].split('_')[0] == 'true' for i in l]
    else:
        return [i.split(code + '-')[1].split('_')[0] for i in l]

def process_one(data_file, type_exp):
    data_ = pd.read_csv(data_file)
    #type_exp = op.basename(data_file).split('.')[0]
    data = data_[['Participant Private ID', 'Zone Type', 'Response', 'Correct',
                  'res_1.1' if 'exp1' in type_exp else 'res_2.1',
                  'res_1.2' if 'exp1' in type_exp else 'res_2.2',
                  'reconstructions_1' if 'exp1' in type_exp else 'reconstructions_2',
                  'ans_1' if 'exp1' in type_exp else 'ans_2']] # also save resp1 and resp2 for permutation tests?
    data = data[data['Zone Type'] == 'response_button_text']
    if 'reconstructions_1' in data.columns.to_list():
        data = data.rename(columns={'reconstructions_1': 'reconstructions', 'ans_1': 'ans',
                                    'res_1.1': 'option1', 'res_1.2': 'option2'})
    else:
        data = data.rename(columns={'reconstructions_2': 'reconstructions', 'ans_2': 'ans',
                                    'res_2.1': 'option1', 'res_2.2': 'option2'})

    codes = ['sub', 'mod', 'opt', 'recon']
    out = { k:[] for k in codes}
    for code in codes:
        out[code] = parse_info(data['reconstructions'].to_list(), code) # remove empty strings

    out['target_response'] = data['ans'].to_list()
    #out['observed_response'] = data['Response'].to_list()
    out['observed_response'] = [random.sample(sublist, 1)[0] for sublist in zip(data['option1'].values, data['option2'].values)]
    out['accuracy'] = (out['observed_response'] == data['ans'].values).astype(float)
    out['file'] = [str(int(i)) for i in data['Participant Private ID'].values]

    assert len(set([len(out[k])  for k in out.keys()])) == 1, 'Unequal number of elements in out dict'
    out_df = pd.DataFrame(out)
    #out_df['file'] = op.basename(data_file).split('.')[0]
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
            out_df= out_df.append(process_one(f, args.type_exp))

        o = True
        r = True
        for s in [str(i) for i in range(1, 6)]:
            for m in ['mlp', 'densenet', 'seq2seq']:
                for d in out_df['file'].unique():
                    temp = out_df[(out_df['sub']==s) & (out_df['mod']==m) & (out_df['file']==d) & (out_df['opt']==o) & (out_df['recon']==r)]
                    res = res.append(pd.DataFrame({'sub': [s], 'mod':[m], 'opt': [o], 'recon': [r],
                                                   'accuracy':[temp['accuracy'].mean()],
                                                   'file': [d]}))

                temp = res[(res['sub'] == s) & (res['mod'] ==m) & (res['opt'] == o) & (res['recon'] == r)]
                res_perm = res_perm.append(pd.DataFrame({'sub': [s], 'mod': [m], 'opt': [o], 'recon': [r],
                                               'accuracy': [temp['accuracy'].mean()],
                                               'perm': [p]}))

        #out_df['perm'] = p
        res['perm'] = p
        #out_df_perm = out_df_perm.append(out_df)
        res_all_files = res_all_files.append(res)

    res_perm = res_perm.sort_values(by=['sub'])
    res_all_files = res_all_files.sort_values(by=['sub'])

    plotdir = args.data_dir.replace('data', 'pics')
    if not op.isdir(plotdir): makedirs(plotdir)
    #
    # for s in res['file'].unique():
    #     plot_dot_acc(res[res['file']==s], title=op.basename(s).split('.')[0], plotdir=plotdir)

    plot_dot_acc(res_perm, title='all_dots_perm', plot_box=True, plotdir=plotdir)

    if not op.isdir(args.save_dir): makedirs(args.save_dir)
    res_all_files.to_csv(op.join(args.save_dir, 'results_avg_over_12words_perm.csv'))
    res_perm.to_csv(op.join(args.save_dir, 'results_avg_over_12words_avg_over_subjs_perm.csv'))


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process exp 1 or 2')
    parser.add_argument('--data_dir',  type=str, help='Path to the dir with spreadsheets for exp 1 or 2')
    parser.add_argument('--save_dir', type=str, help='Path to results')
    parser.add_argument('--n_perms', '-n', type=int, help='Number of permutations', default=1000)

    args = parser.parse_args()
    args.type_exp = 'exp1'
    run_main(args)
