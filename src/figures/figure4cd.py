'''
plot VAD and STOI results with permutation baseline

python figures/figure4cd.py \
    --task jip_janneke \
    --type_perm shift \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --n_folds 12

python figures/figure4cd.py \
    --task jip_janneke \
    --metric stoi \
    --type_perm shuffle \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --n_folds 12
'''

import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')


import os.path as op
import argparse
import pandas as pd
import json
import seaborn as sns

from scipy.stats import ttest_rel
from utils.wilcoxon import wilcoxon
from utils.private.datasets import get_subjects_by_code
from matplotlib import pyplot as plt
from utils.general import get_stat_pval
from figures.figure4b import plot_metric_boxplot

def main(args):
    for type_perm in args.type_perm:
        file_tag = '_shuf_words' if type_perm == 'shuffle' else '_sil_only'

        if type_perm == 'shuffle' and 'vad' in args.metric:
            raise ValueError

        for metric in args.metric:
            results = pd.DataFrame()
            for subject in args.subject:
                print(subject)
                for model in args.model:
                    gen_dir = op.join(args.res_dir, args.task, subject, model)
                    best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']

                    for itrial, trial in enumerate([0, best_trial]):  # optimized and non-optimized
                        trial_dir = op.join(gen_dir, str(trial), 'eval')
                        a = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + metric + '.csv'),
                                        index_col=[0]).reset_index(drop=True)
                        a['subject'] = subject
                        a['model'] = model
                        a['trial'] = 'non-optimized' if trial == 0 else 'optimized'
                        #a['successful'] = b['successful'].values if metric == 'stoi' else True # remove same or just all 1e-5?
                        a['successful'] = ~(a['stoi']==1e-5).values if metric == 'stoi' else True
                        results = results.append(a)

                        # UNCOMMENT THIS WHEN PERMUTATIONS ARE DONE
                        if itrial == 1:  # best_trial
                            b = pd.read_csv(
                                op.join(trial_dir,
                                        'eval_n_folds' + str(args.n_folds) + '_' + metric + '_audio_ceil.csv'),
                                index_col=[0])
                            b['subject'] = subject
                            b['model'] = model
                            b['trial'] = 'ceiling'
                            b['successful'] = ~(b['stoi'] == 1e-5).values if metric == 'stoi' else True
                            results = results.append(pd.DataFrame(b))

                            b = pd.read_csv(
                                op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_' + metric + '_perm'+file_tag+'.csv'),
                                index_col=[0])
                            b['subject'] = subject
                            b['model'] = model
                            b['trial'] = 'permutation'
                            b['successful'] = ~(b['stoi'] == 1e-5).values if metric == 'stoi' else True
                            results = results.append(pd.DataFrame(b))

            # plot boxplots per metric
            results_all = results.copy()
            results = results[results['successful']==True]
            plot_metric_boxplot(metric='vad_match' if metric == 'vad' else 'stoi',
                                data=results[results['trial'] == 'optimized'],
                                floors=results[results['trial'] == 'permutation'],
                                ceilings=results[results['trial']=='ceiling'],
                                title='fig4cd_eval_' + metric + '_with_perm' + file_tag,
                                ylim=(0, 1.1) if metric == 'vad' else (-.25, 1),
                                plot_dots=args.plot_dots,
                                plotdir=args.plot_dir)

            # UNCOMMENT THIS WHEN PERMUTATIONS ARE DONE
            # significance
            print(type_perm)
            print(metric)
            m = 'stoi' if metric == 'stoi' else 'vad_match'
            for subject in args.subject:
                for model in args.model:
                    baseline = results[(results['subject'] == subject) &
                                            (results['model'] == model) &
                                            (results['trial'] == 'permutation')][m].values
                    val = results[(results['trial'] == 'optimized') &
                                       (results['subject'] == subject) &
                                       (results['model'] == model)][m].mean()
                    print(subject + ' & ' + model + ' pval: ' + str(get_stat_pval(val, baseline)))

            # optimized vs non-optimized effects:
            col = 'vad_match' if metric == 'vad' else 'stoi'
            w = wilcoxon(results_all[results_all['trial'] == 'optimized'][col] -
                     results_all[results_all['trial'] == 'non-optimized'][col], alternative='greater',
                     zero_method='zsplit')
            print(metric, 'overall effect: ', w)
            # vad
            # overall effect:  WilcoxonResult(w_statistic=8981.0, z_statistic=1.194230269167449, pvalue=0.11619395285700229)

            r1 = results_all[results_all['trial'] == 'optimized'].reset_index(drop=True)
            r2 = results_all[results_all['trial'] == 'non-optimized'].reset_index(drop=True)
            ind1 = r1[r1[col] == 1e-5].index
            ind2 = r2[r2[col] == 1e-5].index
            r1 = r1.drop(ind1.union(ind2))
            r2 = r2.drop(ind1.union(ind2))
            w = wilcoxon(r1[col] -  r2[col], alternative='greater', zero_method='zsplit')
            print(metric, 'overall effect: ', w)
            for model in args.model:
                # w = wilcoxon(results_all[(results_all['trial'] == 'optimized') & (results_all['model']==model)][col] -
                #          results_all[(results_all['trial'] == 'non-optimized') & (results_all['model']==model)][col], alternative='greater',
                #          zero_method='zsplit')
                w = wilcoxon(r1[r1['model']==model][col] - r2[r2['model']==model][col],
                             alternative='greater',  zero_method='zsplit')
                # w = wilcoxon(results[(results['trial'] == 'optimized') & (results['model']==model)][col] -
                #          results[(results['trial'] == 'non-optimized') & (results['model']==model)][col], alternative='greater',
                #          zero_method='zsplit')
                print(metric, model, w)
            # vad
            # mlp
            # WilcoxonResult(w_statistic=1035.0, z_statistic=0.8834103134391214, pvalue=0.18850731022792244)
            # densenet
            # WilcoxonResult(w_statistic=844.5, z_statistic=-0.5189982852996928, pvalue=0.698119031503621)
            # seq2seq
            # WilcoxonResult(w_statistic=1168.5, z_statistic=1.866216929729946, pvalue=0.03100551070358523)

            # stoi
            # overall effect:  WilcoxonResult(w_statistic=1596470.0, z_statistic=6.225375184648896, pvalue=2.4020248999154475e-10)
            # mlp
            # WilcoxonResult(w_statistic=19502.0, z_statistic=0.8316505708726653, pvalue=0.20280310423574938)
            # densenet
            # WilcoxonResult(w_statistic=22005.0, z_statistic=4.959327023105429, pvalue=3.536890085663792e-07)
            # seq2seq
            # WilcoxonResult(w_statistic=17990.0, z_statistic=0.5118439259143766, pvalue=0.3043801226042436)

            # for results_all (keep paired folds even if they are = 1e-05)
            # stoi
            # overall effect: WilcoxonResult(w_statistic=9345.0, z_statistic=1.7141973829208899, pvalue=0.04324624069370467)
            # mlp
            # WilcoxonResult(w_statistic=966.5, z_statistic=0.37912255458376526, pvalue=0.3522984291037323)
            # densenet
            # WilcoxonResult(w_statistic=1250.0, z_statistic=2.4661704181871325, pvalue=0.0068283171224118285)
            # seq2seq
            # WilcoxonResult(w_statistic=889.0, z_statistic=-0.1914042712622849, pvalue=0.5758955630261474)

            # removing all folds where either optimized or non-optimized is 1e-5
            # stoi
            # overall effect:  WilcoxonResult(w_statistic=8423.0, z_statistic=2.883136913928876, pvalue=0.001968681326600156)
            # mlp
            # WilcoxonResult(w_statistic=925.0, z_statistic=1.2986749734081604, pvalue=0.09702774813907777)
            # densenet
            # WilcoxonResult(w_statistic=1085.0, z_statistic=3.271096527708001, pvalue=0.0005356566834592478)
            # seq2seq
            # WilcoxonResult(w_statistic=787.0, z_statistic=0.14243531966412082, pvalue=0.4433680828609898)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--metric', '-x', type=str,  nargs="+",
                        choices=['vad', 'stoi'],
                        default=['vad', 'stoi'],
                        help='Metric to use')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Subject to run')
    parser.add_argument('--type_perm',  type=str,  nargs="+",
                        choices=['shift', 'shuffle'],
                        default=['shift', 'shuffle'],
                        help='Model to run')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--res_dir', '-r', type=str, help='Results directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
    parser.add_argument('--plot_dots', dest='plot_dots', action='store_true')
    parser.add_argument('--no-plot_dots', dest='plot_dots', action='store_false')
    parser.set_defaults(plot_dots=True)

    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)
