'''
plot word classification results: from audio reconstructions and raw brain data with permutation baseline



!!!!
- still need to replace bootstrapping with averaging over folds per permutation
- add best mlp result per subject (out of 3 feature input types)
- check if a new way to get permutations for SVM and MLP needs adjustments here: now 120 perms per dir now
!!!!


python figures/figure5a.py \
    --task jip_janneke \
    --gen_res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/ \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --clf_type logreg \
    --n_folds 12
'''

import pandas as pd
import json
import argparse
from matplotlib import pyplot as plt
import seaborn as sns
import os.path as op
import glob
from os import makedirs
from utils.private.datasets import get_subjects_by_code

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def plot_classify_boxplot(data, floors=None, title='', plot_dots=False, plotdir=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    if floors is not None:
        sns.boxplot(x='subject', y='accuracy', data=floors,
                    hue='case',
                    boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
                    whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
                    medianprops=dict(linestyle=':', linewidth=.5, color='black'),
                    capprops=dict(linestyle=':', linewidth=.5, color='black'),
                    orient='v',
                    ax=ax, showfliers=False, whis=[5, 95])  # added showfliers = False, whis=[5, 95]
    if plot_dots:
        sns.stripplot(x='subject', y='accuracy', data=data,
                      hue='case', size=3, orient='v',
                      jitter=.2, dodge=True, alpha=.4, linewidth=.6,
                      edgecolors='black', ax=ax)
    sns.boxplot(x='subject', y='accuracy', data=data,
                hue='case',
                boxprops=dict(alpha=.7),
                medianprops=dict(color='red'),
                ax=ax, orient='v', showfliers=False, whis=[5, 95])  # added showfliers = False, whis=[5, 95]

    plt.title(title)
    plt.ylim(-.1, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    if plotdir != '':
        plt.savefig(op.join(plotdir, title + '.pdf'),
                    dpi=160,
                    transparent=True)
        plt.close()


def main(args):
    params = {}
    params['sr'] = [15, 25, 75]
    params['input_band'] = ['70-170', '60-300']
    params['input_ref'] = ['car', 'bipolar']
    if not op.exists(args.plot_dir): makedirs(args.plot_dir)

    for metric in args.metric:
        results = pd.DataFrame()
        for subject in args.subject:
            print(subject)
            for input in ['reconstructed', 'brain_input']: # have to keep outside otherwise in plots: mlp, brain12, densenet, seq2seq
                for model in args.model:
                    gen_dir = op.join(args.gen_res_dir, 'optuna', args.task, subject, model)
                    trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
                    trial_dir = op.join(gen_dir, str(trial), 'eval')

                    a = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + metric + '_' + args.clf_type + '.csv'), index_col=[0])
                    b = {}
                    b['input'] = input
                    b['accuracy'] = a[a['input'] == input]['accuracy'].mean()
                    b['subject'] = [subject]
                    b['model'] = [model]
                    b['cv'] = [0]
                    b['trial'] = 'non-optimized' if trial == 0 else 'optimized'
                    b['dir'] = gen_dir
                    b['clf'] = args.clf_type
                    results = results.append(pd.DataFrame(b))

                    # add permitations
                    # if itrial == 1: # best_trial
                    #     a = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + metric + '_perm.csv'),
                    #                     index_col=[0])
                    #     for perm in range(1000):
                    #     #for boot in range(100): # replace this with averaging over folds per permutation
                    #         b = {}
                    #         b['input'] = input
                    #         b['accuracy'] = a[(a['input'] == input) & (a['perm']==perm)]['accuracy'].mean()
                    #         #b['accuracy'] = a[a['input'] == input]['accuracy'].sample(n=100).mean()
                    #         b['subject'] = [subject]
                    #         b['model'] = [model]
                    #         b['cv'] = [0]
                    #         b['trial'] = 'permutation'
                    #         b['dir'] = gen_dir
                    #         results = results.append(pd.DataFrame(b))

        # plot boxplot group per subject and model
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        sns.boxplot(x='model', y='accuracy', data=results,
                    hue='input', hue_order=['brain_input', 'reconstructed'],
                    boxprops=dict(alpha=.7, linewidth=1), width=.9,
                    whiskerprops=dict(linewidth=1),
                    medianprops=dict(color='red', linewidth=1),
                    capprops=dict(linewidth=.5),
                    ax=ax, orient='v', showfliers=False, whis=[5, 95]) # added showfliers = False, whis=[5, 95]

        sns.stripplot(x='model', y='accuracy', data=results,
                      hue='input', hue_order=['brain_input', 'reconstructed'],  size=9, orient='v',
                      jitter=.2, dodge=True, alpha=.4, linewidth=.6, linestyle='-',
                      edgecolors='black', ax=ax)


        for s in [0, 2, 4]:
            for i, j in zip(ax.get_children()[s].get_offsets().data, ax.get_children()[s+1].get_offsets().data):
                ax.plot([i[0], j[0]], [i[1], j[1]], color="black", alpha=0.1)

        #
        # sns.boxplot(x='subject', y='clf', data=results[results['input'] == 'brain_input'],
        #             hue='model', hue_order=['mlp', 'densenet', 'seq2seq'], palette='pastel',
        #             boxprops=dict(alpha=.7, linewidth=1), width=.9,
        #             whiskerprops=dict(linewidth=1),
        #             medianprops=dict(color='red', linewidth=1),
        #             capprops=dict(linewidth=.5),
        #             ax=ax, orient='v', showfliers=False, whis=[5, 95]) # added showfliers = False, whis=[5, 95]

        # plt.ylim(-.03, 1.03)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.legend([], [], frameon=False)

        if args.plot_dir != '':
            plt.savefig(op.join(args.plot_dir, 'fig5a_eval_optimized_' + metric + '_recon_brain_' + args.clf_type + '.pdf'), dpi=160,
                        transparent=True)
            plt.close()

        # # for trial in ['optimized', 'non-optimized']:
        # # results = results.sort_values(by='model', ascending=False) differnt sortings for temp and perm
        # temp = results[results['trial'].isin(['optimized'])].copy()
        # temp.loc[(temp['input'] == 'reconstructed')&(temp['trial'] == 'optimized')&(temp['model'] == 'mlp'),'case'] = 'mlp'
        # temp.loc[(temp['input'] == 'reconstructed')&(temp['trial'] == 'optimized')&(temp['model'] == 'densenet'),'case'] = 'densenet'
        # temp.loc[(temp['input'] == 'reconstructed')&(temp['trial'] == 'optimized')&(temp['model'] == 'seq2seq'),'case'] = 'seq2seq'
        # temp.loc[(temp['input'] == 'brain_input') & (temp['cv'] == 1) & (temp['clf'] == 'mlp') & (temp['model'] == 'default') & (temp['trial'] == 'matched'), 'case'] = 'brain_default_cv'
        # temp.loc[(temp['input'] == 'brain_input') & (temp['cv'] == 1) & (temp['clf'] == 'mlp') & (temp['model'] == 'avg_time') & (temp['trial'] == 'matched'), 'case'] = 'brain_avg_time_cv'
        # temp.loc[(temp['input'] == 'brain_input') & (temp['cv'] == 1) & (temp['clf'] == 'mlp') & (temp['model'] == 'avg_chan') & (temp['trial'] == 'matched'), 'case'] = 'brain_avg_chan_cv'
        # temp.loc[(temp['input'] == 'brain_input') & (temp['cv'] == 1) & (temp['clf'] == 'svm') & (temp['model'] == 'default') & (temp['trial'] == 'matched'), 'case'] = 'brain_default_cv_svm'
        # temp.loc[(temp['input'] == 'brain_input') & (temp['cv'] == 0) & (temp['trial'] == 'optimized'), 'case'] = 'brain12'
        #
        # perm = results[results['trial'].isin(['permutation'])].copy()
        # perm.loc[(perm['input'] == 'reconstructed')&(perm['trial'] == 'permutation')&(perm['model'] == 'mlp'),'case'] = 'mlp'
        # perm.loc[(perm['input'] == 'reconstructed')&(perm['trial'] == 'permutation')&(perm['model'] == 'densenet'),'case'] = 'densenet'
        # perm.loc[(perm['input'] == 'reconstructed')&(perm['trial'] == 'permutation')&(perm['model'] == 'seq2seq'),'case'] = 'seq2seq'
        # perm.loc[(perm['input'] == 'brain_input') & (perm['cv'] == 1) & (perm['clf'] == 'mlp') & (perm['model'] == 'default') & (perm['trial'] == 'permutation'), 'case'] = 'brain_default_cv'
        # perm.loc[(perm['input'] == 'brain_input') & (perm['cv'] == 1) & (perm['clf'] == 'mlp') & (perm['model'] == 'avg_time') & (perm['trial'] == 'permutation'), 'case'] = 'brain_avg_time_cv'
        # perm.loc[(perm['input'] == 'brain_input') & (perm['cv'] == 1) & (perm['clf'] == 'mlp') & (perm['model'] == 'avg_chan') & (perm['trial'] == 'permutation'), 'case'] = 'brain_avg_chan_cv'
        # perm.loc[(perm['input'] == 'brain_input') & (perm['cv'] == 1) & (perm['clf'] == 'svm') & (perm['model'] == 'default') & (perm['trial'] == 'permutation'), 'case'] = 'brain_default_cv_svm'
        # perm.loc[(perm['input'] == 'brain_input') & (perm['cv'] == 0) & (perm['trial'] == 'permutation'), 'case'] = 'brain12'
        #
        # c = sns.color_palette()
        # c_ = [c[i] for i in [0, 1, 2, 4, 4, 3, 3, 3]]
        #
        # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        #
        # sns.boxplot(x='subject', y='accuracy', data=temp,
        #             hue='case', palette=c_,
        #             boxprops=dict(alpha=.7, linewidth=1), width=.9,
        #             whiskerprops=dict(linewidth=1),
        #             medianprops=dict(color='red', linewidth=1),
        #             capprops=dict(linewidth=.5),
        #             ax=ax, orient='v', showfliers=False, whis=[5, 95]) # added showfliers = False, whis=[5, 95]
        # sns.boxplot(x='subject', y='accuracy', data=perm,
        #             hue='case', width=.9,
        #             boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
        #             whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
        #             medianprops=dict(linestyle=':', linewidth=.5, color='black'),
        #             capprops=dict(linestyle=':', linewidth=.5, color='black'),
        #             orient='v',
        #             ax=ax, color='gray', showfliers=False, whis=[5, 95]) # added showfliers = False, whis=[5, 95]
        # sns.stripplot(x='subject', y='accuracy', data=temp,
        #               hue='case', size=3, orient='v', palette=c_,
        #               jitter=.2, dodge=True, alpha=.4, linewidth=.6,
        #               edgecolors='none', ax=ax)





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
    parser.add_argument('--clf_feat', '-f', type=str,  nargs="+",
                        choices=['default', 'avg_time', 'avg_chan'],
                        default=['default', 'avg_time', 'avg_chan'],
                        help='Input classifier features')
    parser.add_argument('--metric', '-x', type=str,
                        choices=['classify'],
                        default=['classify'],
                        help='Metric to use')
    parser.add_argument('--clf_type', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='logreg',
                        help='Type of classifier')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--gen_res_dir', '-r', type=str, help='Results directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')

    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)

#
#
# '''
# plot word classification results: from audio reconstructions and raw brain data with permutation baseline
#
#
#
# !!!!
# - still need to replace bootstrapping with averaging over folds per permutation
# - add best mlp result per subject (out of 3 feature input types)
# - check if a new way to get permutations for SVM and MLP needs adjustments here: now 120 perms per dir now
# !!!!
#
#
# python figures/figure5a_old.py \
#     --task jip_janneke \
#     --gen_res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/ \
#     --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
#     --n_folds 12
# '''
#
# import pandas as pd
# import json
# import argparse
# from matplotlib import pyplot as plt
# import seaborn as sns
# import os.path as op
# import glob
# from os import makedirs
#
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
# def plot_classify_boxplot(data, floors=None, title='', plot_dots=False, plotdir=None):
#     fig, ax = plt.subplots(1, 1, figsize=(6, 3))
#
#     if floors is not None:
#         sns.boxplot(x='subject', y='accuracy', data=floors,
#                     hue='case',
#                     boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
#                     whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
#                     medianprops=dict(linestyle=':', linewidth=.5, color='black'),
#                     capprops=dict(linestyle=':', linewidth=.5, color='black'),
#                     orient='v',
#                     ax=ax, showfliers=False, whis=[5, 95])  # added showfliers = False, whis=[5, 95]
#     if plot_dots:
#         sns.stripplot(x='subject', y='accuracy', data=data,
#                       hue='case', size=3, orient='v',
#                       jitter=.2, dodge=True, alpha=.4, linewidth=.6,
#                       edgecolors='black', ax=ax)
#     sns.boxplot(x='subject', y='accuracy', data=data,
#                 hue='case',
#                 boxprops=dict(alpha=.7),
#                 medianprops=dict(color='red'),
#                 ax=ax, orient='v', showfliers=False, whis=[5, 95])  # added showfliers = False, whis=[5, 95]
#
#     plt.title(title)
#     plt.ylim(-.1, 1.1)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
#     if plotdir != '':
#         plt.savefig(op.join(plotdir, title + '.pdf'),
#                     dpi=160,
#                     transparent=True)
#         plt.close()
#
#
# def main(args):
#     params = {}
#     params['sr'] = [15, 25, 75]
#     params['input_band'] = ['70-170', '60-300']
#     params['input_ref'] = ['car', 'bipolar']
#     if not op.exists(args.plot_dir): makedirs(args.plot_dir)
#
#     for metric in args.metric:
#         results = pd.DataFrame()
#         for subject in args.subject:
#             print(subject)
#             for input in ['reconstructed', 'brain_input']: # have to keep outside otherwise in plots: mlp, brain12, densenet, seq2seq
#                 for model in args.model:
#                     gen_dir = op.join(args.gen_res_dir, 'optuna', args.task, subject, model)
#                     best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))
#
#                     for itrial, trial in enumerate([0, best_trial['_number']]):  # optimized and non-optimized
#                         trial_dir = op.join(gen_dir, str(trial), 'eval')
#                         a = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + metric + '.csv'), index_col=[0])
#                         b = {}
#                         b['input'] = input
#                         b['accuracy'] = a[a['input'] == input]['accuracy'].mean()
#                         b['subject'] = [subject]
#                         b['model'] = [model]
#                         b['cv'] = [0]
#                         b['trial'] = 'non-optimized' if trial == 0 else 'optimized'
#                         b['dir'] = gen_dir
#                         results = results.append(pd.DataFrame(b))
#                         if itrial == 1: # best_trial
#                             a = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + metric + '_perm.csv'),
#                                             index_col=[0])
#                             for perm in range(1000):
#                             #for boot in range(100): # replace this with averaging over folds per permutation
#                                 b = {}
#                                 b['input'] = input
#                                 b['accuracy'] = a[(a['input'] == input) & (a['perm']==perm)]['accuracy'].mean()
#                                 #b['accuracy'] = a[a['input'] == input]['accuracy'].sample(n=100).mean()
#                                 b['subject'] = [subject]
#                                 b['model'] = [model]
#                                 b['cv'] = [0]
#                                 b['trial'] = 'permutation'
#                                 b['dir'] = gen_dir
#                                 results = results.append(pd.DataFrame(b))
#
#             for f in ['svm_word_id', 'svm_word_id_perm', 'mlp_word_id', 'mlp_word_id_perm']:
#                 gen_dir = op.join(args.gen_res_dir, 'brain_input_decoding', args.task, subject, f)
#                 for clf_feat in args.clf_feat:
#                     all_dirs = glob.glob(op.join(gen_dir, clf_feat + '_*'))
#                     assert len(all_dirs) == 12, 'number of dirs is ' + str(len(all_dirs))
#                     for d in all_dirs:
#                         b = {}
#                         if f in ['svm_word_id', 'mlp_word_id']:
#                             a = pd.read_csv(op.join(d, 'loo_classify.csv'), index_col=[0])
#                             b['trial'] = 'matched'
#                         else:
#                             a = pd.read_csv(op.join(d, 'perm_classify.csv'), index_col=[0])
#                             b['trial'] = 'permutation'
#                         b['input'] = 'brain_input'
#                         b['accuracy'] = a['accuracy'].mean()
#                         b['subject'] = [subject]
#                         b['model'] = [clf_feat]
#                         b['cv'] = [1]
#                         b['dir'] = d
#                         b['clf'] = 'svm' if 'svm' in f else 'mlp'
#                         results = results.append(pd.DataFrame(b))
#
#         # plot boxplot group per subject and model
#         # for trial in ['optimized', 'non-optimized']:
#         # results = results.sort_values(by='model', ascending=False) differnt sortings for temp and perm
#         temp = results[results['trial'].isin(['optimized', 'matched'])].copy()
#         temp.loc[(temp['input'] == 'reconstructed')&(temp['trial'] == 'optimized')&(temp['model'] == 'mlp'),'case'] = 'mlp'
#         temp.loc[(temp['input'] == 'reconstructed')&(temp['trial'] == 'optimized')&(temp['model'] == 'densenet'),'case'] = 'densenet'
#         temp.loc[(temp['input'] == 'reconstructed')&(temp['trial'] == 'optimized')&(temp['model'] == 'seq2seq'),'case'] = 'seq2seq'
#         temp.loc[(temp['input'] == 'brain_input') & (temp['cv'] == 1) & (temp['clf'] == 'mlp') & (temp['model'] == 'default') & (temp['trial'] == 'matched'), 'case'] = 'brain_default_cv'
#         temp.loc[(temp['input'] == 'brain_input') & (temp['cv'] == 1) & (temp['clf'] == 'mlp') & (temp['model'] == 'avg_time') & (temp['trial'] == 'matched'), 'case'] = 'brain_avg_time_cv'
#         temp.loc[(temp['input'] == 'brain_input') & (temp['cv'] == 1) & (temp['clf'] == 'mlp') & (temp['model'] == 'avg_chan') & (temp['trial'] == 'matched'), 'case'] = 'brain_avg_chan_cv'
#         temp.loc[(temp['input'] == 'brain_input') & (temp['cv'] == 1) & (temp['clf'] == 'svm') & (temp['model'] == 'default') & (temp['trial'] == 'matched'), 'case'] = 'brain_default_cv_svm'
#         temp.loc[(temp['input'] == 'brain_input') & (temp['cv'] == 0) & (temp['trial'] == 'optimized'), 'case'] = 'brain12'
#
#         perm = results[results['trial'].isin(['permutation'])].copy()
#         perm.loc[(perm['input'] == 'reconstructed')&(perm['trial'] == 'permutation')&(perm['model'] == 'mlp'),'case'] = 'mlp'
#         perm.loc[(perm['input'] == 'reconstructed')&(perm['trial'] == 'permutation')&(perm['model'] == 'densenet'),'case'] = 'densenet'
#         perm.loc[(perm['input'] == 'reconstructed')&(perm['trial'] == 'permutation')&(perm['model'] == 'seq2seq'),'case'] = 'seq2seq'
#         perm.loc[(perm['input'] == 'brain_input') & (perm['cv'] == 1) & (perm['clf'] == 'mlp') & (perm['model'] == 'default') & (perm['trial'] == 'permutation'), 'case'] = 'brain_default_cv'
#         perm.loc[(perm['input'] == 'brain_input') & (perm['cv'] == 1) & (perm['clf'] == 'mlp') & (perm['model'] == 'avg_time') & (perm['trial'] == 'permutation'), 'case'] = 'brain_avg_time_cv'
#         perm.loc[(perm['input'] == 'brain_input') & (perm['cv'] == 1) & (perm['clf'] == 'mlp') & (perm['model'] == 'avg_chan') & (perm['trial'] == 'permutation'), 'case'] = 'brain_avg_chan_cv'
#         perm.loc[(perm['input'] == 'brain_input') & (perm['cv'] == 1) & (perm['clf'] == 'svm') & (perm['model'] == 'default') & (perm['trial'] == 'permutation'), 'case'] = 'brain_default_cv_svm'
#         perm.loc[(perm['input'] == 'brain_input') & (perm['cv'] == 0) & (perm['trial'] == 'permutation'), 'case'] = 'brain12'
#
#         c = sns.color_palette()
#         c_ = [c[i] for i in [0, 1, 2, 4, 4, 3, 3, 3]]
#
#         fig, ax = plt.subplots(1, 1, figsize=(8, 4))
#
#         sns.boxplot(x='subject', y='accuracy', data=temp,
#                     hue='case', palette=c_,
#                     boxprops=dict(alpha=.7, linewidth=1), width=.9,
#                     whiskerprops=dict(linewidth=1),
#                     medianprops=dict(color='red', linewidth=1),
#                     capprops=dict(linewidth=.5),
#                     ax=ax, orient='v', showfliers=False, whis=[5, 95]) # added showfliers = False, whis=[5, 95]
#         sns.boxplot(x='subject', y='accuracy', data=perm,
#                     hue='case', width=.9,
#                     boxprops=dict(alpha=.3, facecolor='lightgray', linewidth=.5),
#                     whiskerprops=dict(linestyle=':', linewidth=.5, color='black'),
#                     medianprops=dict(linestyle=':', linewidth=.5, color='black'),
#                     capprops=dict(linestyle=':', linewidth=.5, color='black'),
#                     orient='v',
#                     ax=ax, color='gray', showfliers=False, whis=[5, 95]) # added showfliers = False, whis=[5, 95]
#         sns.stripplot(x='subject', y='accuracy', data=temp,
#                       hue='case', size=3, orient='v', palette=c_,
#                       jitter=.2, dodge=True, alpha=.4, linewidth=.6,
#                       edgecolors='none', ax=ax)
#
#         plt.ylim(-.03, 1.03)
#         plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#         #plt.legend([], [], frameon=False)
#
#         if args.plot_dir != '':
#             plt.savefig(op.join(args.plot_dir, 'fig5a_eval_optimized_' + metric + '_recon_brain_compared_with_perm.pdf'), dpi=160,
#                         transparent=True)
#             plt.close()
#
#
#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='Parameters for sound generation')
#     parser.add_argument('--task', '-t', type=str)
#     parser.add_argument('--model', '-m', type=str,  nargs="+",
#                         choices=['mlp', 'densenet', 'seq2seq'],
#                         default=['mlp', 'densenet', 'seq2seq'],
#                         help='Subject to run')
#     parser.add_argument('--clf_feat', '-f', type=str,  nargs="+",
#                         choices=['default', 'avg_time', 'avg_chan'],
#                         default=['default', 'avg_time', 'avg_chan'],
#                         help='Input classifier features')
#     parser.add_argument('--metric', '-x', type=str,  nargs="+",
#                         choices=['classify'],
#                         default=['classify'],
#                         help='Metric to use')
#     parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
#     parser.add_argument('--gen_res_dir', '-r', type=str, help='Results directory', default='')
#     parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
#
#     args = parser.parse_args()
#
#     main(args)
