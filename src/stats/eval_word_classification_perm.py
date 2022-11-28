'''
Run correlations for permutations of 12 folds in test
Produces distributions very similar to actual correlations because
    - this is based on 1-second fragments (for word classification) with a long silence bit at the end of almost every trial
    - we mostly pick up on speech vs non-speech rather than detailed word profiles
    plus
    - permutations of 12 usually end up with several items in their right matching place

Only for linear SVM classifier

python stats/eval_word_classification_perm.py\
    --task jip_janneke \
    --subject_code xoop \
    --model seq2seq \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --clf_type logreg \
    --n_folds 12 \
    --n_perms 1000

'''

import sys
sys.path.insert(0, '.')

import os
import os.path as op
import argparse
import pandas as pd
import numpy as np
import json

from utils.private.datasets import get_subjects_by_code
from utils.arguments import load_params
from utils.datasets import BrainDataset, split_data, load_data, get_moments
from utils.plots import plot_eval_metric_box
from evaluate_decoders.eval_word_classification import classify_words


def main(args):
    for isubj, subject in enumerate(args.subject):
        print(subject)
        for imod, model in enumerate(args.model):
            gen_dir = op.join(args.res_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
            trial_dir = op.join(gen_dir, str(best_trial), 'eval')
            args.model_path = op.join(gen_dir, str(best_trial),
                                      op.basename([f.path for f in os.scandir(op.join(gen_dir, str(best_trial))) if f.is_dir() and 'eval' not in f.path][0]))
            args = load_params(args)
            word_len_sampl = args.sr

            res_clf_perm = pd.DataFrame()

            for perm in range(args.n_perms):
                # select a test fold (validation word) at random
                print(perm)
                # fold = np.random.randint(0, 12) # run over folds, take mean
                for fold in range(args.n_folds):  # take mean over all folds

                    fold_dir = op.join(trial_dir, 'fold' + str(fold))

                    # get correct filename for fragments_for_classif = from the fold number + subsets_full_for_classif
                    word_fragments_fold = op.join(fold_dir, op.basename(args.subsets_path).replace('.csv', '_fold'+str(fold)+'.csv'))
                    fragments_for_classif_fold = op.join(fold_dir, 'fragments_fold'+str(fold)+'_full_for_classif.csv')
                    word_fragments = pd.read_csv(word_fragments_fold, index_col=0, header=0)

                    # load data corresponding to the train-test split in fragments_for_classif (including momemts)
                    dataset = BrainDataset(args.input_file, args.output_file, fragments_for_classif_fold,
                                           args.input_sr, args.output_sr, args.fragment_len, args.input_delay)
                    trainset, valset, testset = split_data(dataset)
                    x_mean, x_std, t_mean, t_std, pca = get_moments(trainset, args.input_mean, args.input_std,
                                                                    args.output_mean, args.output_std, args.use_pca,
                                                                    args.n_pcs, can_write=False)
                    train_loader, val_loader, test_loader = load_data(trainset, valset, testset,
                                                                      batch_size=len(trainset),
                                                                      shuffle_train=False)
                    _, train_targets, _ = next(iter(train_loader))
                    val_predictions = np.load(os.path.join(fold_dir, 'val_predictions.npy'))
                    val_targets = np.load(os.path.join(fold_dir, 'val_targets.npy'))
                    train_targets = train_targets.detach().numpy().reshape(-1, word_len_sampl, train_targets.shape[-1])

                    # shuffle assignments in word_fragments from subsets_path_fold: fold_folder + subsets_path + foldX.csv
                    word_fragments.loc[word_fragments['subset']=='train','text'] = \
                    word_fragments[word_fragments['subset']=='train']['text'].values[np.random.permutation(word_fragments.shape[0]-1)]
                    # word_fragments['text'] = word_fragments['text'].values[np.random.permutation(word_fragments.shape[0])]

                    # run classifier
                    res_clf_recon = classify_words(word_fragments,
                                                   train_targets, val_targets, val_predictions,
                                                   t_mean, t_std,
                                                   clf_type=args.clf_type)
                    res_clf_recon['perm'] = perm
                    res_clf_recon['fold'] = fold

                    # repeat for brain input data
                    dataset_ = BrainDataset(args.input_file, args.input_file, fragments_for_classif_fold,
                                            args.input_sr, args.input_sr, args.fragment_len, args.input_delay)
                    trainset_, valset_, testset_ = split_data(dataset_)
                    train_loader_, val_loader_, test_loader_ = load_data(trainset_, valset_, testset_,
                                                                         batch_size=len(trainset_),
                                                                         shuffle_train=False)
                    _, train_input, _ = next(iter(train_loader_))
                    _, val_input, _ = next(iter(val_loader_))
                    train_input = train_input.detach().numpy().reshape(-1, word_len_sampl, train_input.shape[-1])
                    val_input = val_input.detach().numpy().reshape(-1, word_len_sampl, val_input.shape[-1])
                    res_clf_brain = classify_words(word_fragments,
                                                   train_input, val_input, val_input,
                                                   x_mean, x_std,
                                                   clf_type=args.clf_type)
                    res_clf_brain = res_clf_brain[1:]
                    res_clf_brain['input'] = 'brain_input'
                    res_clf_brain['perm'] = perm
                    res_clf_brain['fold'] = fold
                    res_clf_perm = res_clf_perm.append(res_clf_recon, ignore_index=True)
                    res_clf_perm = res_clf_perm.append(res_clf_brain, ignore_index=True)
                print('perm done')

            plotdir = trial_dir.replace('results', 'pics/decoding')
            res_clf_perm.to_csv(os.path.join(trial_dir, 'eval_n_folds12_classify_' + args.clf_type + '_perm.csv'))
            #plot_eval_metric_box('pearsonr_perm', res_clf_perm, plotdir=plotdir)
            plot_eval_metric_box('classify_perm_' + args.clf_type, res_clf_perm[['input', 'accuracy']], plotdir=plotdir, by='input')



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
    parser.add_argument('--clf_type', '-c', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='svm_linear',
                        help='Type of classifier')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--n_perms', '-p', type=int, help='Number of permutations', default=1000)
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)
