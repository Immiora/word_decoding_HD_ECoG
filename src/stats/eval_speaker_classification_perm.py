'''
Run speaker classification only on test samples but use all audio data for training:
    train on ~110 per subject
    test on 12 per subject, LOO-CV (12 x 5 folds)

Word decoding just used as sanity check for data at some point, can be removed later

Two ways to run speaker decoding: with denormalized speech reconstructions (this really helps use t_mean and t_std to differentiate
between subjects). Produces 100% on speech reconstructions.

Alternatively, run on normalized speech reconstructions, but accuracy suffers a lot on optuna_v1 (with 20 mel as min),
maybe will improve with 40 mel as min

For now run more conceptually correct version: normalized (no extra help from scaling/centering with t_mean and t_std per subejct)

To use denormalized version adjust this:

    # in classify_speakers():
    # normalize over subjects here: compute on train, apply to validation (targets and prediction)
    t_mean = Tensor(np.mean(x_train.reshape((-1, x_train.shape[-1])), 0))
    t_std = Tensor(np.std(x_train.reshape((-1, x_train.shape[-1])), 0))

    # in model_pass():
    # predictions.append(
        denormalize(out.detach(), t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy().squeeze())
    targets.append(
        denormalize(preprocess_t(t, t_mean, t_std, args.clip_t_value, device),
                    t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy().squeeze())

    # in main():
    # t_mean_ = Tensor(np.zeros((train_targets.shape[-1])))
    # t_std_ = Tensor(np.ones((train_targets.shape[-1])))
    # temp = classify_words(word_fragments_fold, train_targets, val_targets, val_predictions, t_mean_, t_std_)
    temp = classify_words(word_fragments_fold, train_targets, val_targets, val_predictions, t_mean, t_std)



python stats/eval_speaker_classification_perm.py \
    --task jip_janneke \
    --model densenet \
    --clf_type logreg \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
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
import random

from models.select_model import select_model
from utils.training import *
from utils.arguments import load_params
from utils.datasets import BrainDataset, split_data, load_data, get_moments
from utils.training import preprocess_t
from utils.plots import plot_eval_metric_box
from utils.private.datasets import get_subjects_by_code
from evaluate_decoders.eval_speaker_classification import resample_audio, classify_speakers

def main(args):

    def model_pass(target_loader):
        print('Computing predictions')
        predictions, targets = [], []
        for (x, t, ith) in target_loader:
            out = model(preprocess_x(x, x_mean, x_std, args.clip_x_value, device, pca=None))
            # predictions.append(
            #     denormalize(out.detach(), t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy().squeeze())
            # targets.append(
            #     denormalize(preprocess_t(t, t_mean, t_std, args.clip_t_value, device),
            #                 t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy().squeeze())

            # since I do not normalize again during classificaiton, keep data normalized
            predictions.append(out.cpu().detach().numpy().squeeze())
            targets.append(preprocess_t(t, t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy().squeeze())
        return np.array(predictions), np.array(targets)

    subjects = args.subject
    for clf in args.clf_type:
        print(clf)

        for imod, xmodel in enumerate(args.model):
            res_clf = pd.DataFrame()
            train_targets_c = {k:[] for k in subjects}
            val_targets_c =  {k:[] for k in subjects}
            val_predictions_c =  {k:[] for k in subjects}

            for isubj, subject in enumerate(subjects):
                print(subject)
                gen_dir = op.join(args.res_dir, args.task, subject, xmodel)
                best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
                trial_dir = op.join(gen_dir, str(best_trial), 'eval')

                for fold in range(args.n_folds):
                    fold_dir = op.join(trial_dir, 'fold' + str(fold))
                    assert op.exists(fold_dir), 'No fold dir at ' + fold_dir
                    args.model_path = op.join(fold_dir,
                                              op.basename([f.path for f in os.scandir(op.join(gen_dir, str(best_trial))) if
                                                           f.is_dir() and 'eval' not in f.path][0]))
                    args = load_params(args)
                    word_len_sampl = args.input_sr

                    # get correct filename for fragments_for_classif = from the fold number + subsets_full_for_classif
                    fragments_for_classif_fold = op.join(fold_dir, 'fragments_fold' + str(fold) + '_full_for_classif.csv')

                    dataset = BrainDataset(args.input_file, args.output_file, fragments_for_classif_fold,
                                           args.input_sr, args.output_sr, args.fragment_len, args.input_delay)
                    trainset, valset, testset = split_data(dataset)
                    x_mean, x_std, t_mean, t_std, pca = get_moments(trainset, args.input_mean, args.input_std,
                                                                    args.output_mean, args.output_std, args.use_pca,
                                                                    args.n_pcs, can_write=False)

                    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
                    if torch.cuda.is_available() and args.use_cuda:
                        torch.backends.cudnn.deterministic = True
                        torch.cuda.set_device(device)
                    train_loader, val_loader, test_loader = load_data(trainset, valset, testset,
                                                                      batch_size=word_len_sampl,
                                                                      shuffle_train=False)
                    # set up models
                    model = select_model(args, train_loader, device)
                    model_path = os.path.join(args.model_path, 'checkpoint_499.pth')
                    assert os.path.exists(model_path), 'Model path does not exist'
                    saved = torch.load(model_path, map_location=torch.device(device))
                    model.load_state_dict(saved['model_state_dict'])
                    model = model.eval()

                    # pass through the model: prediction are audio reconstructions, targets are target audio
                    train_predictions, train_targets = model_pass(train_loader)
                    val_predictions, val_targets = model_pass(val_loader)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    train_targets_c[subject].append(resample_audio(train_targets))
                    val_targets_c[subject].append(resample_audio(val_targets))
                    val_predictions_c[subject].append(resample_audio(val_predictions))
                    print(len(train_targets_c[subject]))

            for perm in range(args.n_perms):
                test_speaker = random.sample(subjects, 1)[0]
                print(perm)
                temp = {'accuracy': [], 'input': ['target_audio', 'reconstructed'], 'perm': [perm]*2}
                acc1, acc2 = [], []
                for c in range(args.n_folds): # take mean over all folds
                #c = np.random.randint(0, args.n_folds)
                    speakers_ = {'id':[], 'subset':[]}
                    train_targets = []
                    for subject in subjects:
                        speakers_['subset'].extend(['train'] * train_targets_c[subject][c].shape[0])
                        if subject == test_speaker:
                            train_targets.append(train_targets_c[subject][c])
                        else:
                            train_targets.append(train_targets_c[subject][0])
                        speakers_['id'].extend([subject]*train_targets_c[subject][c].shape[0])
                    train_targets = np.concatenate(train_targets)
                    val_targets = val_targets_c[test_speaker][c]
                    val_predictions = val_predictions_c[test_speaker][c]

                    speakers_['id'].extend([test_speaker])
                    speakers_['subset'].extend(['validation'])
                    speakers_ = pd.DataFrame(speakers_)
                    speakers_.loc[speakers_['subset']=='train','id'] = \
                    speakers_[speakers_['subset']=='train']['id'].values[np.random.permutation(speakers_.shape[0]-1)]

                    # double-check this
                    res = pd.DataFrame(classify_speakers(speakers_, train_targets, val_targets, val_predictions,
                                                         clf_type=clf))
                    acc1.append(res[res['input']=='target_audio']['accuracy'].values[0])
                    acc2.append(res[res['input'] == 'reconstructed']['accuracy'].values[0])
                temp['accuracy'].extend([np.mean(acc1), np.mean(acc2)])
                res_clf = res_clf.append(pd.DataFrame(temp), ignore_index=True)
                print(res_clf.tail(2))


            plot_dir = os.path.join(args.res_dir,  args.task).replace('results', 'pics/decoding')
            plot_eval_metric_box('eval_' + xmodel + '_classify_speakers_' + clf + '_perm', res_clf[['input','accuracy']], plotdir=plot_dir, by='input')
            res_clf.to_csv(os.path.join(args.res_dir,  args.task, 'eval_' + xmodel + '_classify_speakers_' + clf + '_perm.csv'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['seq2seq', 'mlp', 'densenet'],
                        default=['seq2seq', 'mlp', 'densenet'],
                        help='Model to run')
    parser.add_argument('--clf_type', '-c', type=str,  nargs="+",
                        choices=['svm_linear', 'mlp', 'logreg'],
                        default=['svm_linear', 'mlp', 'logreg'],
                        help='Type of classifier')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--n_perms', '-p', type=int, help='Number of permutations', default=1000)
    parser.add_argument('--res_dir', '-o', type=str, help='Result output directory', default='')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)
