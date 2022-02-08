''''
Plot results of the noise study

python figures/figure5c.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --noise_sigma 0.1 .5 1 2 \
    --clf_type logreg
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
from utils.plots import get_model_cmap

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def main(args):
    results = pd.DataFrame()

    for noise_sigma in args.noise_sigma:
        for imod, model in enumerate(args.model):
            for isubj, subject in enumerate(args.subject):
                print(subject)
                for input in ['reconstructed', 'brain_input']:
                    gen_dir = op.join(args.res_dir, args.task, subject, model)
                    best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
                    trial_dir = op.join(gen_dir, str(best_trial), 'eval')
                    a = pd.read_csv(op.join(trial_dir, 'eval_n_folds12_classify_' + args.clf_type + '_noise' + str(noise_sigma) + '.csv'), index_col=0, header=0)
                    b = {}
                    b['input'] = input
                    b['accuracy'] = a[a['input'] == input]['accuracy'].mean()
                    b['subject'] = [subject]
                    b['model'] = [model]
                    b['noise'] = [noise_sigma]
                    results = results.append(pd.DataFrame(b))



    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    sns.stripplot(x='noise', y='accuracy', data=results,
                  hue='input', hue_order=['brain_input', 'reconstructed'], size=9, orient='v',
                  jitter=.2, dodge=True, alpha=.4, linewidth=.6,
                  edgecolors='black', ax=ax)

    sns.boxplot(x='noise', y='accuracy', data=results,
                hue='input', hue_order=['brain_input', 'reconstructed'],
                boxprops=dict(alpha=.7, facecolor='none'),
                medianprops=dict(color='red'),
                ax=ax, orient='v', showfliers=False, whis=[5, 95])  # added showfliers = False, whis=[5, 95]

    markers = ['*', '^', 'o', 's', 'd']
    colors = sns.color_palette()[:3]
    n = len(args.subject)
    for sub, mar in zip(range(n), markers):
        for box in range(8):
            pos = ax.get_children()[box].get_offsets().data
            for i, col in zip(range(3), colors):
                marsize = 11 if mar == '*' else 7 # was 11 and 15: too big
                ax.plot(pos[sub+n*i][0], pos[sub+n*i][1], markersize=marsize, markeredgecolor='black', color=col, marker=mar)

    if args.plot_dir is not None:
        plt.savefig(op.join(args.plot_dir,
                                 'fig5e_classify_word_noise_study_' + args.clf_type + '_brain_recon_with_markers.pdf'),
                    dpi=160, transparent=True)
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
    parser.add_argument('--noise_sigma', type=float,  nargs="+")
    parser.add_argument('--clf_type', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='logreg',
                        help='Type of classifier')
    parser.add_argument('--res_dir', '-r', type=str, help='Results directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')

    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)

#
# '''
# plot speaker classification results: from audio reconstructions and audio data with permutation baseline
#
# python figures/figure5c_old.py \
#     --task jip_janneke \
#     --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
#     --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
#     --clf_type logreg \
#     --n_folds 12
# '''
#
# import sys
# sys.path.insert(0, '.')
#
# import pandas as pd
# import argparse
# import os.path as op
#
# from os import makedirs
# from utils.general import get_stat_pval
# from figures.figure5a import plot_classify_boxplot
#
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
#
#
# def main(args):
#
#     if not op.exists(args.plot_dir): makedirs(args.plot_dir)
#
#     for metric in args.metric:
#         results = pd.DataFrame()
#         gen_dir = op.join(args.res_dir, args.task)
#
#         for input in ['reconstructed', 'target_audio']: # have to keep outside otherwise in plots: mlp, brain12, densenet, seq2seq
#
#             for model in args.model:
#                 for clf in args.clf_type:
#                     a = pd.read_csv(op.join(gen_dir, 'eval_' + model + '_' + metric + '_' + clf + '.csv'), index_col=[0])
#                     b = {}
#                     b['input'] = input
#                     b['accuracy'] = a[a['input'] == input]['accuracy'].mean()
#                     b['model'] = [model if input == 'reconstructed' else 'audio']
#                     b['trial'] = 'real'
#                     b['case'] = clf
#                     results = results.append(pd.DataFrame(b))
#
#                     # UNCOMMENT TO ADD PERM
#                     # a = pd.read_csv(op.join(gen_dir, 'eval_' + model + '_' + metric + '_' + clf + '_perm.csv'), index_col=[0])
#                     # b = {}
#                     # #for boot in range(100):
#                     # for perm in range(1000):
#                     #     b['input'] = input
#                     #     #b['accuracy'] = a[a['input'] == input]['accuracy'].sample(n=100).mean()
#                     #     b['accuracy'] = a[(a['input'] == input) & (a['perm'] == perm)]['accuracy'].mean() # already means, but keep this for consistency
#                     #     b['model'] = [model if input == 'reconstructed' else 'audio']
#                     #     b['trial'] = 'permutation'
#                     #     b['case'] = clf
#                     #     results = results.append(pd.DataFrame(b))
#
#         #
#         temp = results.copy()
#         temp['subject'] = temp['model']
#         temp = temp.drop(columns=['model'])
#         plot_classify_boxplot(data=temp[temp['trial']=='real'],
#                               floors=temp[temp['trial']=='permutation'],
#                               title='fig5c_eval_' + metric + '_' + clf + '_recon_audio',
#                               plot_dots=args.plot_dots,
#                               plotdir=args.plot_dir)
#
#         for clf in args.clf:
#             for model in args.model:
#                 for input in ['reconstructed']:
#                     baseline = results[(results['case']==clf) & (results['input']==input) &
#                                             (results['model']==model) &
#                                             (results['trial']=='permutation')]['accuracy'].values
#                     val = results[(results['case'] == clf) & (results['input']==input) &
#                                        (results['trial']=='real') &
#                                        (results['model'] == model)]['accuracy'].mean()
#                     print(clf + ' & ' + model + ' & ' + input + ' pval: ' + str(get_stat_pval(val, baseline)))
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
#                         choices=['classify_speakers'],
#                         default=['classify_speakers'],
#                         help='Metric to use')
#     parser.add_argument('--clf_type', '-c', type=str,  nargs="+",
#                         choices=['svm', 'mlp', 'logreg'],
#                         default=['svm', 'mlp', 'logreg'],
#                         help='Type of classifier')
#     parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
#     parser.add_argument('--res_dir', '-r', type=str, help='Results directory', default='')
#     parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
#     parser.add_argument('--plot_dots', dest='plot_dots', action='store_true')
#     parser.add_argument('--no-plot_dots', dest='plot_dots', action='store_false')
#     parser.set_defaults(plot_dots=False)
#
#     args = parser.parse_args()
#
#     main(args)
