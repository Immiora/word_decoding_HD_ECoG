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



python evaluate_decoders/eval_speaker_classification.py \
    --task jip_janneke \
    --clf_type logreg \
    --model mlp densenet \
    --trial 0 \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
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
import librosa
import random

from models.select_model import select_model
from utils.training import *
from utils.arguments import load_params
from utils.datasets import BrainDataset, split_data, load_data, get_moments
from utils.private.datasets import get_subjects_by_code
from torch import Tensor
from utils.training import preprocess_t
from sklearn.svm import SVC
from utils.plots import plot_eval_metric_box
from evaluate_decoders.eval_word_classification import classify_words
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def resample_audio(x, out_time=15, out_feat=20):
    x_n, x_time, x_feat= x.shape
    out = []
    for i in range(x_n):
        out.append(librosa.resample(librosa.resample(x[i].T,  x_time, out_time, res_type='scipy').T, x_feat, out_feat, res_type='scipy'))
    return np.array(out)

def classify_speakers(speakers, x_train, x_val, pred_val, t_mean=None, t_std=None, clf_type='svm_linear', return_proba=False):

    if 'svm' in clf_type:
        if '_' in clf_type:
            clf_type, kernel = clf_type.split('_')
        else:
            kernel = 'linear'
    else:
        kernel = None

    labels = pd.factorize(speakers['id'])[0]
    val_labels = labels[speakers[speakers['subset']=='validation'].index]
    train_labels = np.delete(labels, speakers[speakers['subset']=='validation'].index)

    # normalize over subjects here: compute on train, apply to validation (targets and prediciton)
    #t_mean = Tensor(np.mean(x_train.reshape((-1, x_train.shape[-1])), 0))
    #t_std = Tensor(np.std(x_train.reshape((-1, x_train.shape[-1])), 0))

    # normalize and reshape
    if t_mean is not None and t_std is not None:
        x_train = preprocess_t(Tensor(x_train), t_mean, t_std, clip_t_value=None, DEVICE='cpu').detach().numpy()
        x_val = preprocess_t(Tensor(x_val), t_mean, t_std, clip_t_value=None, DEVICE='cpu').detach().numpy()
        pred_val = preprocess_t(Tensor(pred_val), t_mean, t_std, clip_t_value=None, DEVICE='cpu').detach().numpy()
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)
    pred_val = pred_val.reshape(pred_val.shape[0], -1)

    # train classifier
    # seed = np.random.randint(0, 1000, 1)[0]
    # if not nonlin:
    #     clf = SVC(random_state=seed, C=1)
    # else:
    #     clf = MLPClassifier(random_state=seed, hidden_layer_sizes=50)
    seed = np.random.randint(0, 1000, 1)[0]
    if clf_type == 'svm':
        clf = SVC(random_state=seed, C=1, kernel=kernel, probability=return_proba)
    elif clf_type == 'logreg':
        clf = LogisticRegression(random_state=seed, C=1, solver='saga')
    elif clf_type == 'mlp':
        clf = MLPClassifier(random_state=seed, hidden_layer_sizes=50)
    else:
        raise ValueError
    clf.fit(x_train, train_labels)

    # test on targets
    acc_targets = clf.score(x_val, val_labels)

    # test on predictions
    acc_predictions = clf.score(pred_val, val_labels)

    # prepare output
    out = {'input':['target_audio', 'reconstructed'],
           'target_label': [speakers.loc[labels==val_labels]['id'].values[0]] * 2,
           'target_label_id': list(val_labels) * 2,
           'predicted_label': [speakers.loc[labels==clf.predict(x_val)]['id'].values[0],
                                speakers.loc[labels == clf.predict(pred_val)]['id'].values[0]],
           'predicted_label_id': list(clf.predict(x_val)) + list(clf.predict(pred_val)),
           'accuracy': [acc_targets, acc_predictions]}
    #return out
    if not return_proba:
        return out
    else:
        return out, clf.predict_proba(pred_val), val_labels


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

    subject_codes = ['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop']
    subjects = get_subjects_by_code(subject_codes)
    for clf in args.clf_type:

        for imod, xmodel in enumerate(args.model):
            res_clf = pd.DataFrame()
            prob = np.zeros((args.n_folds*len(subjects), len(subjects)))  # fold x unique classes
            val_label = np.zeros((args.n_folds*len(subjects),))  # target labels per fold
            res_word = pd.DataFrame()
            train_targets_c = {k:[] for k in subjects}
            val_targets_c =  {k:[] for k in subjects}
            val_predictions_c =  {k:[] for k in subjects}

            for isubj, subject in enumerate(subjects):
                print(subject)
                gen_dir = op.join(args.res_dir, args.task, subject, xmodel)
                #best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']

                if args.trial == 'best':
                    best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
                else:
                    try:
                        best_trial = int(args.trial)
                    except:
                        raise ValueError

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
                    word_fragments = pd.read_csv(args.subsets_path, index_col=0, header=0)

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

                    # UNCOMMENT FOR TESTING
                    # # check that word decoding also works on these data
                    # word_fragments_fold_file = op.join(fold_dir,
                    #                               op.basename(args.subsets_path).replace('.csv', '_fold' + str(fold) + '.csv'))
                    # word_fragments_fold = pd.read_csv(word_fragments_fold_file, index_col=0, header=0)
                    # t_mean_ = Tensor(np.zeros((train_targets.shape[-1])))
                    # t_std_ = Tensor(np.ones((train_targets.shape[-1])))
                    # temp = classify_words(word_fragments_fold, train_targets, val_targets, val_predictions, t_mean_, t_std_)
                    #
                    # # if denormalize in model_pass:
                    # # temp = classify_words(word_fragments_fold, train_targets, val_targets, val_predictions, t_mean, t_std)
                    # temp['subject'] = subject
                    # res_word = res_word.append(temp, ignore_index=True)

            counter = 0
            for i, test_speaker in enumerate(subjects):
                print('test subject: ' + test_speaker)
                for c in range(args.n_folds):
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

                    # sanity check: use another subject's validation predictions or random (just rely on t_mean and t_std)
                    #subjects2 = subjects.copy()
                    #subjects2.remove(test_speaker)
                    #val_predictions = val_predictions_c[random.sample(subjects2, 1)[0]][c]
                    #val_predictions = np.random.random(size=val_targets.shape)

                    speakers_['id'].extend([test_speaker])
                    speakers_['subset'].extend(['validation'])
                    res, prob[counter], val_label[counter]  = classify_speakers(pd.DataFrame(speakers_),
                                            train_targets, val_targets, val_predictions,
                                            clf_type=clf, return_proba=True)
                    res['fold'] = [i*10 + c] * 2
                    res_clf = res_clf.append(pd.DataFrame(res), ignore_index=True)
                    print(res_clf.tail(1))
                    counter += 1

            labels, uniques = pd.factorize(speakers_['id'])
            plot_dir = os.path.join(args.res_dir,  args.task).replace('results', 'pics/decoding')
            plot_eval_metric_box('eval_' + xmodel + '_classify_speakers_trial' + args.trial + '_' + clf,
                                 res_clf[['input','accuracy']], plotdir=plot_dir, by='input')
            res_clf.to_csv(os.path.join(args.res_dir,  args.task, 'eval_' + xmodel +
                                        '_classify_speakers_trial' + args.trial + '_' + clf + '.csv'))

            x = pd.DataFrame(prob, columns=[i for i in uniques])
            x.insert(0, 'target', [uniques[int(i)] for i in val_label])
            x.to_csv(os.path.join(args.res_dir,  args.task, 'eval_' + xmodel +
                                  '_classify_speakers_trial' + args.trial + '_' + clf + '_prob.csv'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['seq2seq', 'mlp', 'densenet'],
                        default=['seq2seq', 'mlp', 'densenet'],
                        help='Model to run')
    parser.add_argument('--trial', type=str,
                        choices=['best', '0'],
                        default='best',
                        help='Optimized (best) or non-optimized recon (0)')
    parser.add_argument('--clf_type', '-c', type=str,  nargs="+",
                        choices=['svm_linear', 'mlp', 'logreg'],
                        default=['svm_linear', 'mlp', 'logreg'],
                        help='Type of classifier')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of CV folds')
    parser.add_argument('--res_dir', '-o', type=str, help='Result output directory', default='')
    args = parser.parse_args()

    main(args)


# '''
# Run speaker classification only on test samples: 12 per subject, train on 59, test on one, LOO-CV
#
#
# python evaluate_decoders/eval_speaker_classification.py \
#     --task jip_janneke \
#     --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
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
# import librosa
#
# from torch import Tensor
# from utils.training import preprocess_t
# from sklearn.svm import SVC
# from utils.plots import plot_eval_metric_box
# from stats.eval_pearsonr_stoi_perm_shuffle import load_t_moments, normalize
#
#
#
# def classify_speakers(speakers, x_train, x_val, pred_val, t_mean=None, t_std=None):
#     labels = speakers['id'].values
#     val_labels = labels[speakers[speakers['subset']=='validation'].index]
#     train_labels = np.delete(labels, speakers[speakers['subset']=='validation'].index)
#
#     # normalize and reshape
#     if t_mean is not None and t_std is not None:
#         x_train = preprocess_t(Tensor(x_train), t_mean, t_std, clip_t_value=None, DEVICE='cpu').detach().numpy()
#         x_val = preprocess_t(Tensor(x_val), t_mean, t_std, clip_t_value=None, DEVICE='cpu').detach().numpy()
#         pred_val = preprocess_t(Tensor(pred_val), t_mean, t_std, clip_t_value=None, DEVICE='cpu').detach().numpy()
#     x_train = x_train.reshape(x_train.shape[0], -1)
#     x_val = x_val.reshape(x_val.shape[0], -1)
#     pred_val = pred_val.reshape(pred_val.shape[0], -1)
#
#     # train classifier
#     seed = np.random.randint(0, 1000, 1)[0]
#     clf = SVC(random_state=seed, C=1)
#     clf.fit(x_train, train_labels)
#
#     # test on targets
#     acc_targets = clf.score(x_val, val_labels)
#
#     # test on predictions
#     acc_predictions = clf.score(pred_val, val_labels)
#
#     # prepare output
#     out = {'input':['target_audio', 'reconstructed'],
#            'target_label': [speakers.loc[labels==val_labels]['id'].values[0]] * 2,
#            'target_label_id': list(val_labels) * 2,
#            'predicted_label': [speakers.loc[labels==clf.predict(x_val)]['id'].values[0],
#                                 speakers.loc[labels == clf.predict(pred_val)]['id'].values[0]],
#            'predicted_label_id': list(clf.predict(x_val)) + list(clf.predict(pred_val)),
#            'accuracy': [acc_targets, acc_predictions]}
#     return out
#
#
# def main(args):
#     subjects = get_subjects_by_code(code_list)
#     for imod, model in enumerate(args.model):
#         res_clf = pd.DataFrame()
#         val_predictions_z, val_targets_z, speakers = [], [], []
#         val_predictions_ori, val_targets_ori =[], []
#
#         for isubj, subject in enumerate(subjects):
#             print(subject)
#             gen_dir = op.join(args.res_dir, args.task, subject, model)
#             best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
#             trial_dir = op.join(gen_dir, str(best_trial), 'eval')
#
#             for fold in range(args.n_folds):
#                 fold_dir = op.join(trial_dir, 'fold' + str(fold))
#                 assert op.exists(fold_dir), 'No fold dir at ' + fold_dir
#
#                 val_predictions = np.load(os.path.join(fold_dir, 'val_predictions.npy'))
#                 val_targets = np.load(os.path.join(fold_dir, 'val_targets.npy'))
#                 t_mean, t_std = load_t_moments(fold_dir)
#                 predictions = librosa.resample(librosa.resample(normalize(val_predictions, t_mean, t_std)[0].T,
#                                                                 val_predictions.shape[1], 15, res_type='scipy').T,
#                                                val_predictions.shape[-1], 20, res_type='scipy')
#                 targets = librosa.resample(librosa.resample(normalize(val_targets, t_mean, t_std)[0].T,
#                                                                 val_targets.shape[1], 15, res_type='scipy').T,
#                                                val_targets.shape[-1], 20, res_type='scipy')
#
#                 val_predictions_ori.append(normalize(val_predictions, t_mean, t_std)[0].T)
#                 val_targets_ori.append(normalize(val_targets, t_mean, t_std)[0].T)
#                 val_predictions_z.append(predictions)
#                 val_targets_z.append(targets)
#                 speakers.append(isubj)
#         val_predictions_z = np.array(val_predictions_z)
#         val_targets_z = np.array(val_targets_z)
#         speakers = pd.DataFrame({'id': speakers})
#
#         for sample in range(speakers.shape[0]):
#             speakers['subset'] = 'train'
#             speakers.loc[sample, 'subset'] = 'validation'
#             train_targets = val_targets_z[speakers[speakers['subset']=='train'].index]
#             val_targets = val_targets_z[speakers[speakers['subset']=='validation'].index]
#             val_predictions = val_predictions_z[speakers[speakers['subset']=='validation'].index]
#             res = classify_speakers(speakers, train_targets, val_targets, val_predictions)
#             res['fold'] = [sample] * 2
#             res_clf = res_clf.append(pd.DataFrame(res), ignore_index=True)
#
#
#
#         plot_eval_metric_box('accuracy', res_clf[['input','accuracy']], plotdir=None, by='input')
#         res_clf.to_csv(os.path.join(args.res_dir,  args.task, 'eval_' + model + '_classify_speakers.csv'))
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
#     parser.add_argument('--res_dir', '-o', type=str, help='Result output directory', default='')
#     args = parser.parse_args()
#
#     main(args)
