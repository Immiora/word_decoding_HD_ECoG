'''
Look at probabilities of the word classifier: any difference between MLP / densenet / seq2seq

python evaluate_decoders/eval_alt_word_classification.py \
    --task jip_janneke \
    --clf_type logreg \
    --subject_code fvxs \
    --trial best \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna


'''

import sys
sys.path.insert(0, '.')

import os
import argparse
import pandas as pd
import numpy as np
import os.path as op
import json

from utils.private.datasets import get_subjects_by_code
from utils.training import *
from utils.arguments import load_params
from utils.datasets import BrainDataset, split_data, load_data, get_moments
from models.select_model import select_model
from matplotlib import pyplot as plt
from evaluate_decoders.eval_word_classification import classify_words
from utils.plots import get_model_cmap


def main(args):

    def model_pass(target_loader):
        print('Computing predictions')
        predictions, targets = [], []
        for (x, t, ith) in target_loader:
            out = model(preprocess_x(x, x_mean, x_std, args.clip_x_value, device, pca=None))
            predictions.append(
                denormalize(out.detach(), t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy().squeeze())
            targets.append(
                denormalize(preprocess_t(t, t_mean, t_std, args.clip_t_value, device),
                            t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy().squeeze())
        return np.array(predictions), np.array(targets)

    for isubj, subject in enumerate(args.subject):
        for imod, mod in enumerate(args.model):
            gen_dir = op.join(args.res_dir, args.task, subject, mod)
            if args.trial == 'best':
                best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
            else:
                try:
                    best_trial = int(args.trial)
                except:
                    raise ValueError

            trial_dir = op.join(gen_dir, str(best_trial), 'eval')
            prob = np.zeros((12, 12)) # fold x unique classes
            val_label = np.zeros((12)) # target labels per fold
            prob_brain = np.zeros((12, 12)) # fold x unique classes
            val_label_brain = np.zeros((12)) # target labels per fold
            res = pd.DataFrame()

            for fold in range(12):
                print(fold)
                fold_dir = op.join(trial_dir, 'fold' + str(fold))
                args.model_path = op.join(fold_dir,
                                          op.basename([f.path for f in os.scandir(op.join(gen_dir, str(best_trial))) if
                                                       f.is_dir() and 'eval' not in f.path][0]))
                args = load_params(args)
                word_len_sampl = args.input_sr

                # get correct filename for fragments_for_classif = from the fold number + subsets_full_for_classif
                fragments_for_classif_fold = op.join(fold_dir, 'fragments_fold' + str(fold) + '_full_for_classif.csv')
                word_fragments_fold = op.join(fold_dir,
                                              op.basename(args.subsets_path).replace('.csv', '_fold' + str(fold) + '.csv'))
                word_fragments = pd.read_csv(word_fragments_fold, index_col=0, header=0)

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
                print('Device: ', device)
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

                np.save(os.path.join(fold_dir, 'val_predictions'), val_predictions)
                np.save(os.path.join(fold_dir, 'val_targets'), val_targets)

                print('Computing accuracy on word classification')
                res_clf, prob[fold], val_label[fold] = classify_words(word_fragments,
                                                                      train_targets, val_targets, val_predictions,
                                                                      t_mean, t_std,
                                                                      clf_type=args.clf_type,
                                                                      return_proba=True) # changed Jan 29, check

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
                res_clf_brain, prob_brain[fold], val_label_brain[fold] = classify_words(word_fragments,
                                                                                        train_input, val_input, val_input,
                                                                                        x_mean, x_std,
                                                                                        clf_type=args.clf_type,
                                                                                        return_proba=True)
                # save res_clf, res_clf_brain per fold (use same format as before)
                res_clf_brain = res_clf_brain[1:]
                res_clf_brain['input'] = 'brain_input'
                res_clf = res_clf.append(res_clf_brain, ignore_index=True)
                res_clf.to_csv(os.path.join(fold_dir, 'eval_classify_' + args.clf_type + '.csv'))

                # keep results across folds
                res_clf['fold'] = fold
                res = res.append(res_clf)

            # save results across folds:
            res.to_csv(op.join(trial_dir, 'eval_n_folds12_classify_' + args.clf_type + '.csv'))

            # make plots: word classification
            labels, uniques = pd.factorize(word_fragments['text'])
            plotdir = trial_dir.replace('results', 'pics/decoding')
            for i, pr in enumerate([prob, prob_brain]):
                x = pd.DataFrame(pr, columns=[i for i in uniques])
                x.insert(0, 'target', [uniques[int(i)] for i in val_label])
                input_type = '_' if i == 0 else '_brain_'
                x.to_csv(op.join(trial_dir, 'eval_n_folds12_classify_' + args.clf_type + input_type + 'prob.csv'))
            # np.savetxt(op.join(trial_dir, 'eval_n_folds12_classify_prob.csv'), prob)

            plt.figure(figsize=(14, 6))
            for i, (pr, lab) in enumerate(zip([prob, prob_brain], [val_label, val_label_brain])):
                sort_order = [uniques.to_list().index(uniques[int(i)]) for i in lab]
                plt.subplot(1,2,i+1)
                plt.imshow(pr[:,sort_order], vmin=0, vmax=1, cmap=get_model_cmap(mod))
                plt.colorbar()
                plt.xlabel('All words')
                plt.ylabel('Targets (folds)')
                plt.yticks(range(12), [uniques[int(i)] for i in lab])
                plt.xticks(range(12), [uniques[int(i)] for i in lab], rotation=45)
                #plt.xticks(range(12), [i for i in uniques], rotation=45)
                plt.title('left: recon, right: brain')
            if plotdir is not None:
                plt.savefig(os.path.join(plotdir, 'eval_classify_' + args.clf_type + '_trial' + args.trial + '_confusion_matrix_recon_brain_sorted.pdf'),
                                                        dpi=160, transparent=True)
                plt.close()






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop', 'fvxs'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Model to run')
    parser.add_argument('--trial', type=str,
                        choices=['best', '0'],
                        default='best',
                        help='Optimized (best) or non-optimized recon (0)')
    parser.add_argument('--clf_type', '-c', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='svm_linear',
                        help='Type of classifier')
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)
    main(args)
