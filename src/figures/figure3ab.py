'''
plot optimization history

python figures/figure3ab.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures
'''

import argparse
import os.path as op
import pandas as pd
import numpy as np
import json
import os
import seaborn as sns

from matplotlib import pyplot as plt
from utils.private.datasets import get_subjects_by_code

def main(args):
    trials_val = np.zeros((5, 3, 100))
    best_trials_val = np.zeros((5, 3, 100))
    trials_test = np.zeros((5, 3, 2, 12)) # subjects x models x trials x folds (folds)
    medians_test = np.zeros((5, 3, 2))
    #trials_test = np.zeros((5, 3, 11, 12)) # subjects x models x trials x folds (folds)
    #medians_test = np.zeros((5, 3, 11))
    res = {'subject':[], 'model':[], 'trial':[], 'fold':[], 'loss':[]}

    # validation and test
    for isubj, subject in enumerate(args.subject):
        for imod, model in enumerate(args.model):
            gen_dir = op.join(args.res_dir, args.task, subject, model)

            # validation
            temp = pd.read_csv(op.join(gen_dir, 'optuna_trials.tsv'), sep='\t')
            print(temp[temp['state']=='COMPLETE'].shape[0])
            trials_val[isubj, imod] = temp[temp['state']=='COMPLETE']['value'].values
            #best_trials_val[isubj, imod] = np.minimum.accumulate(pd.read_csv(op.join(gen_dir, 'optuna_trials.tsv'), sep='\t')['value'].values)
            best_trials_val[isubj, imod] = np.minimum.accumulate(temp[temp['state']=='COMPLETE']['value'].values)

            # test
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
            #for itrial, trial in enumerate(list(range(10)) + [best_trial]): # optimized and non-optimized
            for itrial, trial in enumerate([0, best_trial]):
                trial_dir = op.join(gen_dir, str(trial), 'eval')
                model_dir = op.basename([f.path for f in os.scandir(op.join(gen_dir, str(trial))) if f.is_dir() and 'eval' not in f.path][0])

                for fold in range(12):
                    fold_dir = op.join(trial_dir, 'fold' + str(fold))
                    assert op.exists(fold_dir), 'No fold dir at ' + fold_dir
                    trials_test[isubj, imod, itrial, fold] = np.load(op.join(fold_dir, model_dir, 'loss_val.npy'))[-1]
                    #if itrial == 0 or itrial == 10:
                    res['subject'].append(subject)
                    res['model'].append(model)
                    res['trial'].append('non-optimized' if trial == 0 else 'optimized')
                    res['fold'].append(fold)
                    res['loss'].append(trials_test[isubj, imod, itrial, fold])
            medians_test[isubj, imod] = np.median(trials_test[isubj,imod],axis=1)

    # validation
    for imod, model in enumerate(args.model):
        plt.figure(figsize=(1.75, 3))
        for isubj, subject in enumerate(args.subject):
            plt.scatter(range(100), trials_val[isubj,imod], alpha=.2)
            plt.plot(range(100), best_trials_val[isubj,imod], label=subject, linewidth=2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylim(0.03, .25)
        plt.title(model)

        plt.savefig(op.join(args.plot_dir, 'fig3a_opt_history_val_5subjs_by_model_'+model+'.pdf'), dpi=160, transparent=True)
        plt.close()

    for isubj, subject in enumerate(args.subject):
        plt.figure(figsize=(1.75, 3))
        for imod, model in enumerate(args.model):
            plt.scatter(range(100), trials_val[isubj, imod], alpha=.2)
            plt.plot(range(100), best_trials_val[isubj, imod], label=model, linewidth=2)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylim(0.03, .25)
        plt.title(subject)

        plt.savefig(op.join(args.plot_dir, 'fig3a_opt_history_val_5subjs_by_subject_'+subject+'.pdf'), dpi=160, transparent=True)
        plt.close()

    # test
    res = pd.DataFrame(res)

    from scipy.stats import ttest_rel
    from utils.wilcoxon import wilcoxon
    a = res[res['trial']=='optimized']['loss'].values
    b = res[res['trial']=='non-optimized']['loss'].values
    print('t-test optimized - non-optimized: ', ttest_rel(b, a))
    w = wilcoxon(b - a, alternative='greater', zero_method='zsplit')
    print(w)

    for imod, model in enumerate(args.model):
        a = res[(res['trial']=='optimized') & (res['model']==model)]['loss'].values
        b = res[(res['trial']=='non-optimized') & (res['model']==model)]['loss'].values
        print(model, 't-test optimized - non-optimized: ', ttest_rel(b, a))
        w = wilcoxon(b - a, alternative='greater', zero_method='zsplit')
        print(w)

    for imod, model in enumerate(args.model):
        if model == 'mlp':
            c = sns.color_palette("Paired")[0:2]
        elif model == 'densenet':
            c = sns.color_palette("Paired")[6:8]
        elif model == 'seq2seq':
            c = sns.color_palette("Paired")[2:4]
        else:
            raise ValueError
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        sns.boxplot(x='subject', y='loss', data=res[res['model']==model], hue='trial',
                    ax=ax, orient='v', palette=c, showfliers=False,
                    boxprops=dict(alpha=.7, linewidth=1), width=.9,
                    whiskerprops=dict(linewidth=1),
                    medianprops=dict(color='red', linewidth=1),
                    capprops=dict(linewidth=.5))
        sns.stripplot(x='subject', y='loss', data=res[res['model']==model], hue='trial',
                      size=3, orient='v', palette=c,
                      jitter=.2, dodge=True, alpha=.4, linewidth=.6,
                      edgecolors='none', ax=ax)
        plt.ylim(0.02, .27)
        plt.title(model)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if args.plot_dir != '':
            plt.savefig(op.join(args.plot_dir, 'fig3b_trial0_vs_best_test_subj_model_'+model+'.pdf'), dpi=160, transparent=True)
            plt.close()

        # for isubj, subject in enumerate(args.subject):
            #sns.swarmplot(data=trials_test[isubj,imod].T, color=colors[isubj], alpha=.2)
            # plt.figure(figsize=(.5, 3))
            # sns.boxplot(data=trials_test[isubj, imod, [0, -1]].T, color=colors[isubj])
            # plt.ylim(0.03, .25)
            # plt.title(model)
            # plt.savefig(op.join(args.plot_dir, 'fig3b_trial0_vs_best_test_subj' +subject+'_model_'+model+'.pdf'), dpi=160, transparent=True)
            # plt.close()

    # from statsmodels.stats.descriptivestats import sign_test
    # sign_test(trials_test[0, imod, 0] - trials_test[0, imod, -1]) : some are significant but would not survive mc


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
                        help='Model to run')
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)
    main(args)