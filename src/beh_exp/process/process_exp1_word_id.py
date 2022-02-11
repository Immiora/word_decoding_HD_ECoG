'''
Process behvaioral data: experiment 1 (word id)

Read in directory with all csv spreadsheets (one per subject)
Process each, populate final csv with results
Save csv with results to plot late in Figure 6a

python beh_exp/process/process_exp1_word_id.py \
    --data_dir /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/subjects/exp1_30subjs \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp1_30subjs
'''

import argparse
import os.path as op
import pandas as pd
import seaborn as sns
import glob
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
        sns.boxplot(x='sub', y='accuracy', data=data[data['recon'] == False], hue='mod', boxprops=dict(alpha=.3),
                    orient='v',
                    ax=ax, color='gray', showfliers=False, whis=[5, 95])
    sns.stripplot(x='sub', y='accuracy', data=data[data['recon'] == True], hue_order=['mlp', 'densenet', 'seq2seq'],
                  hue='mod', size=10, orient='v', jitter=0, dodge=True, alpha=.8, ax=ax)
    sns.stripplot(x='sub', y='accuracy', data=data[data['recon'] == False],
                  hue='mod', size=10, orient='v', jitter=0, dodge=True, color='grey', alpha=0.3, ax=ax)

    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
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
    data = data_[['Participant Private ID', 'Zone Type', 'Response', 'Correct',
                  'reconstructions_1' if 'exp1' in type_exp else 'reconstructions_2',
                  'ans_1' if 'exp1' in type_exp else 'ans_2']] # also save resp1 and resp2 for permutation tests?
    data = data[data['Zone Type'] == 'response_button_text']
    if 'reconstructions_1' in data.columns.to_list():
        data = data.rename(columns={'reconstructions_1': 'reconstructions', 'ans_1': 'ans'})
    else:
        data = data.rename(columns={'reconstructions_2': 'reconstructions', 'ans_2': 'ans'})

    codes = ['sub', 'mod', 'opt', 'recon', 'word']
    out = { k:[] for k in codes}
    for code in codes:
        out[code] = parse_info(data['reconstructions'].to_list(), code) # remove empty strings

    out['target_response'] = data['ans'].to_list()
    out['observed_response'] = data['Response'].to_list()
    out['accuracy'] = data['Correct'].values
    out['file'] = [str(int(i)) for i in data['Participant Private ID'].values]

    assert len(set([len(out[k])  for k in out.keys()])) == 1, 'Unequal number of elements in out dict'
    out_df = pd.DataFrame(out)
    return out_df

def run_main(args):
    out_df = pd.DataFrame()

    for f in glob.glob(op.join(args.data_dir, '*.csv')):
        out_df = out_df.append(process_one(f, args.type_exp))

    # average over 12 words
    res = pd.DataFrame()
    for s in [str(i) for i in range(1, 6)]:
        for m in ['mlp', 'densenet', 'seq2seq']:
            for d in out_df['file'].unique():
                for o in out_df['opt'].unique():
                    for r in out_df['recon'].unique():
                        if r == True:
                            temp = out_df[(out_df['sub']==s) & (out_df['mod']==m) & (out_df['file']==d) & (out_df['opt']==o) & (out_df['recon']==r)]
                            res = res.append(pd.DataFrame({'sub': [s], 'mod':[m], 'opt': [o], 'recon': [r],
                                                           'accuracy':[temp['accuracy'].mean()],
                                                           'file': [d]}))
                        else:
                            temp = out_df[(out_df['sub']==s) & (out_df['mod']=='seq2seq') & (out_df['file']==d) & (out_df['opt']==o) & (out_df['recon']==r)]
                            res = res.append(pd.DataFrame({'sub': [s], 'mod':['seq2seq'], 'opt': [o], 'recon': [r],
                                                           'accuracy':[temp['accuracy'].mean()],
                                                           'file': [d]}))
    # average over 12 words
    res2 = pd.DataFrame()
    for s in [str(i) for i in range(1, 6)]:
        for m in ['mlp', 'densenet', 'seq2seq']:
            for w in out_df['word'].unique(): # changed from target_response, because it is speaker in exp 2
                for o in out_df['opt'].unique():
                    for r in out_df['recon'].unique():
                        if r == True:
                            temp = out_df[(out_df['sub']==s) & (out_df['mod']==m) & (out_df['word']==w) & (out_df['opt']==o) & (out_df['recon']==r)]
                            res2 = res2.append(pd.DataFrame({'sub': [s], 'mod':[m], 'opt': [o], 'recon': [r],
                                                           'accuracy':[temp['accuracy'].mean()],
                                                           'word': [w]}))
                        else:
                            temp = out_df[(out_df['sub']==s) & (out_df['mod']=='seq2seq') & (out_df['word']==w) & (out_df['opt']==o) & (out_df['recon']==r)]
                            res2 = res2.append(pd.DataFrame({'sub': [s], 'mod':['seq2seq'], 'opt': [o], 'recon': [r],
                                                           'accuracy':[temp['accuracy'].mean()],
                                                           'word': [w]}))
    res = res.sort_values(by=['sub'])
    res2 = res2.sort_values(by=['sub', 'mod'])

    plotdir = args.data_dir.replace('data', 'pics')
    if not op.isdir(plotdir): makedirs(plotdir)

    for s in res['file'].unique():
        plot_dot_acc(res[res['file']==s], title=op.basename(s).split('.')[0], plotdir=plotdir)

    plot_dot_acc(res, title='all_dots', plot_box=True, plotdir=plotdir)

    if not op.isdir(args.save_dir): makedirs(args.save_dir)
    out_df.to_csv(op.join(args.save_dir, 'results_all.csv'))
    res.to_csv(op.join(args.save_dir, 'results_avg_over_12words.csv'))
    res2.to_csv(op.join(args.save_dir, 'results_avg_over_subjs.csv'))


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process exp 1 or 2')
    parser.add_argument('--data_dir',  type=str, help='Path to the dir with spreadsheets for exp 1 or 2')
    parser.add_argument('--save_dir', type=str, help='Path to results')

    args = parser.parse_args()
    args.type_exp = 'exp1'
    run_main(args)
