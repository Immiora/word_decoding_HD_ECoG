'''
Load subject's probabilities for word decoding over models, compare

python figures/figure5b.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
    --clf_type logreg \
    --n_folds 12
'''

import sys
sys.path.insert(0, '.')

import os
import os.path as op
import argparse
import pandas as pd
import numpy as np
import json

from scipy.stats import ttest_rel
from utils.wilcoxon import wilcoxon
from utils.private.datasets import get_subjects_by_code
from matplotlib import pyplot as plt
from utils.plots import get_model_cmap


def main(args):

    prob = pd.DataFrame()
    prob_brain = pd.DataFrame()
    maxprob = {key: [] for key in args.model}
    maxprob_brain = {key: [] for key in args.model}

    for imod, model in enumerate(args.model):
        for isubj, subject in enumerate(args.subject):
            print(subject)
            gen_dir = op.join(args.res_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
            trial_dir = op.join(gen_dir, str(best_trial), 'eval')
            temp = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_classify_' + args.clf_type + '_prob.csv'), index_col=0, header=0)
            temp['subject'] = subject
            temp['model'] = model
            prob = prob.append(temp)
            temp = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_classify_' + args.clf_type + '_brain_prob.csv'), index_col=0, header=0)
            temp['subject'] = subject
            temp['model'] = model
            prob_brain = prob_brain.append(temp)

        # plot means
        # for pr, input_type in zip([prob, prob_brain], ['_recon', '_brain']):
            # options = pr.columns.drop(['target', 'subject'])
            # means = pd.DataFrame(columns=options)
            # for target in pr['target'].unique():
            #     means = means.append(pr[(pr['target']==target) & (pr['model']==model)].drop(['subject', 'model'], 1).mean(axis=0), ignore_index=True)
            # means.insert(0, 'target', pr['target'].unique())
            # means = means[np.hstack(['target', means['target'].values])]
            # plt.figure()
            # plt.imshow(means.drop('target', 1), vmin=0, vmax=1, cmap=get_model_cmap(model))
            # plt.colorbar()
            # plt.xlabel('Predicted words')
            # plt.ylabel('Targets (folds)')
            # #plt.xticks(range(12), [i for i in options],  rotation=45)
            # plt.xticks(range(12), [i for i in means['target'].values], rotation=45)
            # plt.yticks(range(12), [i for i in means['target'].values])
            # plt.title(model)
            # if args.plot_dir is not None:
            #     plt.savefig(os.path.join(args.plot_dir,
            #                  'fig5b_eval_classify_prob_matrices_meansubj_' + model + '_' + args.clf_type + input_type + '.pdf'), dpi=160, transparent=True)
            #     plt.close()


        temp = prob[prob['model']==model]
        temp_brain = prob_brain[prob_brain['model']==model]
        for target in temp['target'].unique():
            maxprob[model].extend(temp[temp['target']==target][target])
            maxprob_brain[model].extend(temp_brain[temp_brain['target']==target][target])

    # np.array(maxprob['densenet']).reshape(12, 5)

    print('seq2seq - mlp: ', ttest_rel(maxprob['seq2seq'], maxprob['mlp']))
    w = wilcoxon(np.array(maxprob['seq2seq'])-np.array(maxprob['mlp']), alternative='greater', zero_method='zsplit')
    print(w)
    print('seq2seq - densenet: ', ttest_rel(maxprob['seq2seq'], maxprob['densenet']))
    w = wilcoxon(np.array(maxprob['seq2seq'])-np.array(maxprob['densenet']), alternative='greater', zero_method='zsplit')
    print(w)
    print('densenet - mlp: ', ttest_rel(maxprob['densenet'], maxprob['mlp']))
    w = wilcoxon(np.array(maxprob['densenet'])-np.array(maxprob['mlp']), alternative='greater', zero_method='zsplit')
    print(w)

    print('recon - brain: mlp: ', ttest_rel(maxprob['mlp'], maxprob_brain['mlp']))
    w = wilcoxon(np.array(maxprob['mlp'])-np.array(maxprob_brain['mlp']), alternative='greater', zero_method='zsplit')
    print(w)
    print('recon - brain: densenet: ', ttest_rel(maxprob['densenet'], maxprob_brain['densenet']))
    w = wilcoxon(np.array(maxprob['densenet'])-np.array(maxprob_brain['densenet']), alternative='greater', zero_method='zsplit')
    print(w)
    print('recon - brain: seq2seq: ', ttest_rel(maxprob['seq2seq'], maxprob_brain['seq2seq']))
    w = wilcoxon(np.array(maxprob['seq2seq'])-np.array(maxprob_brain['seq2seq']), alternative='greater', zero_method='zsplit')
    print(w)





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
    parser.add_argument('--clf_type', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='logreg',
                        help='Type of classifier')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)
    main(args)


#
# '''
# Load subject's probabilities for word decoding over models, compare
#
# python figures/figure5b_old.py \
#     --task jip_janneke \
#     --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
#     --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \
#     --n_folds 12
# '''
#
# import sys
# sys.path.insert(0, '.')
#
# import os
# import os.path as op
# import argparse
# import pandas as pd
# import numpy as np
# import json
#
# from matplotlib import pyplot as plt
#
#
#
# def main(args):
#
#     prob = pd.DataFrame()
#     maxprob = {key: [] for key in args.model}
#
#     for imod, model in enumerate(args.model):
#         for isubj, subject in enumerate(args.subject):
#             print(subject)
#             gen_dir = op.join(args.res_dir, args.task, subject, model)
#             best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
#             trial_dir = op.join(gen_dir, str(best_trial), 'eval')
#             temp = pd.read_csv(op.join(trial_dir, 'eval_n_folds' + str(args.n_folds) + '_classify_prob.csv'), index_col=0, header=0)
#             temp['subject'] = subject
#             temp['model'] = model
#             prob = prob.append(temp)
#
#         # plot means
#         options = prob.columns.drop(['target', 'subject'])
#         means = pd.DataFrame(columns=options)
#         for target in prob['target'].unique():
#             means = means.append(prob[(prob['target']==target) & (prob['model']==model)].drop(['subject', 'model'], 1).mean(axis=0), ignore_index=True)
#         means.insert(0, 'target', prob['target'].unique())
#         means = means[np.hstack(['target', means['target'].values])]
#         plt.figure()
#         plt.imshow(means.drop('target', 1), vmin=0, vmax=1, cmap='Oranges')
#         plt.colorbar()
#         plt.xlabel('Predicted words')
#         plt.ylabel('Targets (folds)')
#         #plt.xticks(range(12), [i for i in options],  rotation=45)
#         plt.xticks(range(12), [i for i in means['target'].values], rotation=45)
#         plt.yticks(range(12), [i for i in means['target'].values])
#         plt.title(model)
#         if args.plot_dir is not None:
#             plt.savefig(os.path.join(args.plot_dir, 'fig5b_eval_classify_prob_matrices_meansubj_' + model + '.pdf'), dpi=160, transparent=True)
#             plt.close()
#
#
#         temp = prob[prob['model']==model]
#         for target in temp['target'].unique():
#             maxprob[model].extend(temp[temp['target']==target][target])
#
#     from scipy.stats import ttest_rel
#     from utils.wilcoxon import wilcoxon
#     print('seq2seq - mlp: ', ttest_rel(maxprob['seq2seq'], maxprob['mlp']))
#     w = wilcoxon(np.array(maxprob['seq2seq'])-np.array(maxprob['mlp']), alternative='greater', zero_method='zsplit')
#     print(w)
#     print('seq2seq - densenet: ', ttest_rel(maxprob['seq2seq'], maxprob['densenet']))
#     w = wilcoxon(np.array(maxprob['seq2seq'])-np.array(maxprob['densenet']), alternative='greater', zero_method='zsplit')
#     print(w)
#     print('densenet - mlp: ', ttest_rel(maxprob['densenet'], maxprob['mlp']))
#     w = wilcoxon(np.array(maxprob['densenet'])-np.array(maxprob['mlp']), alternative='greater', zero_method='zsplit')
#     print(w)
#
#
#
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
#                         help='Model to run')
#     parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
#     parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
#     parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
#     args = parser.parse_args()
#
#     main(args)
