'''
Added other classifier options. This needs to be tested if rerunning test + eval


Evaluate audio reconstructions based on brain activity
Use optimal optuna model (pretrained per fold using optuna_test_per_fold.py) to run evaluation per fold
Inputs:
    - model folder path
    - model checkpoint to use
    - fold (for easier parallelization: set up each fold as individual process or per fold optuna_test and optuna_evalute)
    - metric

Several metrics can be used:
    - pearson correlation between reconstructed and target audio across mel features
    - voice activity detection (needs paths to generated waveforms): https://github.com/wiseman/py-webrtcvad
    - STOI (needs paths to generated waveforms): https://github.com/mpariente/pystoi
    - word classification from mel spectrograms

Classify words based on the spectrogram

1. Train on true audio spectrograms
    - train LOO CV on audio spectrograms: need train/test.csv (per fold)
    - take audio from onset + fixed window in time
    - train SVM/CNN
    - compute test accuracy on audio

2. Test on predicted spectrograms: LOO CV:
    - load predictions for the left-out word
    - pass through pretrained classifier
    - compute accuracy
    - repeat for all words

3. Significance testing: parametric / compare with audio / permutations

python ./evaluate_decoders/optuna_evaluate_per_fold.py \
    -p path_to_optuna_$task_$subject_$model_$trial_eval_$foldX_$model_path
    -c checkpoint_499 \
    --metric classify \
    --clf_type svm_linear \
    -s path_to_optuna_$task_$subject_$model_$trial_eval_$foldX



'''

import sys
sys.path.insert(0, '.')

import os
import argparse
import pandas as pd
import numpy as np

from utils.training import *
from utils.plots import plot_scatter_predictions_targets, plot_eval_metric_box
from utils.arguments import load_params
from utils.datasets import BrainDataset, split_data, load_data, get_moments
from models.select_model import select_model
from decimal import Decimal, ROUND_HALF_UP

from evaluate_decoders.eval_pearsonr import calculate_pearsonr
from evaluate_decoders.eval_vad import calculate_vad
from evaluate_decoders.eval_stoi import calculate_stoi
from evaluate_decoders.eval_word_classification import classify_words


def main(args):

    def model_pass(target_loader):
        print('Computing predictions')
        predictions, targets = [], []
        for (x, t, ith) in target_loader:
            out = model(preprocess_x(x, x_mean, x_std, args.clip_x_value, device, pca=None))
            predictions.append(
                denormalize(out.detach(), t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy().squeeze())
            #targets.append(t.transpose(1, 2).cpu().detach().numpy().squeeze())
            targets.append(
                denormalize(preprocess_t(t, t_mean, t_std, args.clip_t_value, device),
                            t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy().squeeze())
        return np.array(predictions), np.array(targets)

    word_len_sampl = args.sr # 1 second window for decoding from word onset, can be changed here

    # get same size window across words + necessary context: contexts_full
    word_fragments = pd.read_csv(args.subsets_path_fold, index_col=0,  header=0)
    contexts_full = pd.read_csv(args.fragments_full, index_col=0,  header=0)
    contexts_full['to_use'] = False

    samples2ind = lambda s: int(Decimal(s).quantize(0, ROUND_HALF_UP))
    step = 1/args.sr
    samples_dur = samples2ind(args.fragment_len / (1 / args.output_sr))
    target_ind = max(samples2ind((samples_dur - 1) / 2), 0)

    for _, row in word_fragments.iterrows():
        onset = row['xmin']
        word_index = (contexts_full[(contexts_full['xmin']<=onset) & (contexts_full['xmax']>=onset)]
                                    ['xmin'] + step*target_ind - onset).abs().idxmin()
        # contexts_full.loc[word_index:word_index + word_len_sampl, 'to_use'] = True # loc is inclusive !
        contexts_full.iloc[word_index:word_index + word_len_sampl, contexts_full.columns.get_loc("to_use")] = True # iloc is more finicky in using labels
        contexts_full.iloc[word_index:word_index + word_len_sampl, contexts_full.columns.get_loc("subset")] = row['subset']

    contexts_full = contexts_full[contexts_full['to_use']==True].reset_index(drop=True)
    fragments_for_classif = args.fragments_full.replace('.csv', '_for_classif.csv')
    contexts_full.to_csv(fragments_for_classif)

    # get data
    dataset = BrainDataset(args.input_file, args.output_file, fragments_for_classif,
                           args.input_sr, args.output_sr, args.fragment_len, args.input_delay)
    trainset, valset, testset = split_data(dataset)

    # compute mean and std on train data
    x_mean, x_std, t_mean, t_std, pca = get_moments(trainset, args.input_mean, args.input_std,
                                                    args.output_mean, args.output_std, args.use_pca,
                                                    args.n_pcs, can_write=False)

    # set up torch
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if torch.cuda.is_available() and args.use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(device)
    print('Device: ', device)

    # data loaders
    train_loader, val_loader, test_loader = load_data(trainset, valset, testset,
                                                      batch_size=word_len_sampl,
                                                      shuffle_train=False)

    # set up models
    model = select_model(args, train_loader, device)
    model_path = os.path.join(args.model_path, args.checkpoint + '.pth')
    assert os.path.exists(model_path), 'Model path does not exist'
    saved = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(saved['model_state_dict'])
    model = model.eval()

    # pass through the model: prediction are audio reconstructions, targets are target audio
    train_predictions, train_targets = model_pass(train_loader)
    val_predictions, val_targets = model_pass(val_loader)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # grab paths to generated audio
    args.audio_val_targets = os.path.join(args.model_path, 'parallel_wavegan', 'targets_validation_gen.wav')
    args.audio_val_predictions = os.path.join(args.model_path, 'parallel_wavegan', args.model_type + '_validation_gen.wav')
    assert os.path.exists(args.audio_val_targets), 'File does not exist: ' + args.audio_val_targets
    assert os.path.exists(args.audio_val_predictions), 'File does not exist: ' + args.audio_val_predictions

    # set up plotting
    plotdir = args.save_dir.replace('results', 'pics/decoding')
    if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
    if not os.path.isdir(plotdir): os.makedirs(plotdir)

    for t, p in zip(val_targets, val_predictions):
        plot_scatter_predictions_targets(t, p, train_tag='val', save_pics=True, plotdir=plotdir)

    # pearson correlation
    if 'pearsonr' in args.metric:
        print('Computing pearsonr')
        res_pearsonr = calculate_pearsonr(val_predictions, val_targets)
        res_pearsonr.index.name = 'mel'
        res_pearsonr.to_csv(os.path.join(args.save_dir, 'eval_pearsonr.csv'))

        plot_eval_metric_box('pearsonr', res_pearsonr, plotdir)

    # VAD
    if 'vad' in args.metric:
        print('Computing VAD match')
        res_vad = calculate_vad(args.audio_val_predictions, args.audio_val_targets)
        res_vad.to_csv(os.path.join(args.save_dir, 'eval_vad.csv'))

        plot_eval_metric_box('vad', res_vad, plotdir)

    # STOI
    if 'stoi' in args.metric:
        print('Computing STOI/ESTOI')
        res_stoi = calculate_stoi(args.audio_val_predictions, args.audio_val_targets)
        res_stoi.to_csv(os.path.join(args.save_dir, 'eval_stoi.csv'))

        plot_eval_metric_box('stoi', res_stoi, plotdir)

    # word classification: train on target audio, test on val target audio and val audio reconstruction
    if 'classify' in args.metric:
        print('Computing accuracy on word classification')
        res_clf = classify_words(word_fragments,
                                 train_targets, val_targets, val_predictions,
                                 t_mean, t_std,
                                 clf_type=args.clf_type)

        # classify same time points: train on brain data, test on brain data
        dataset_ = BrainDataset(args.input_file, args.input_file, fragments_for_classif,
                              args.input_sr, args.input_sr, args.fragment_len, args.input_delay)
        trainset_, valset_, testset_ = split_data(dataset_)
        train_loader_, val_loader_, test_loader_ = load_data(trainset_, valset_, testset_,
                                                          batch_size=len(trainset_),
                                                          shuffle_train=False)
        _, train_input, _ = next(iter(train_loader_))
        _, val_input, _ = next(iter(val_loader_))
        train_input = train_input.detach().numpy().reshape(-1, word_len_sampl, train_input.shape[-1])
        val_input = val_input.detach().numpy().reshape(-1, word_len_sampl, val_input.shape[-1])

        res_clf_input = classify_words(word_fragments,
                                       train_input, val_input, val_input,
                                       x_mean, x_std,
                                       clf_type=args.clf_type)
        res_clf_input = res_clf_input[1:]
        res_clf_input['input'] = 'brain_input'
        res_clf = res_clf.append(res_clf_input, ignore_index=True)
        res_clf.to_csv(os.path.join(args.save_dir, 'eval_classify_' + args.clf_type + '.csv'))

        # make plots: word classification
        plot_eval_metric_box('classify_' + args.clf_type, res_clf[['input','accuracy']], plotdir=plotdir, by='input')


    # save inputs for bookkeeping
    with open(os.path.join(args.save_dir, 'eval_input_parameters.txt'), 'w') as f:
        for key, value in args._get_kwargs():
            f.write(key + '=' + str(value) + '\n')

    # save predictions and targets: switched off temporarily to re-run classifier with logreg
    np.save(os.path.join(args.save_dir, 'val_predictions'), val_predictions)
    np.save(os.path.join(args.save_dir, 'val_targets'), val_targets)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--model_path', '-p', type=str, help='Directory with model')
    parser.add_argument('--checkpoint', '-c', type=str, help='Checkpoint', default='min_val_loss')
    parser.add_argument('--metric', '-m', type=str,  nargs="+",
                        choices=['pearsonr', 'vad', 'stoi', 'classify'],
                        default=['pearsonr', 'vad', 'stoi', 'classify'],
                        help='Metric to use')
    parser.add_argument('--clf_type', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='svm_linear',
                        help='Type of classifier')
    parser.add_argument('--save_dir', '-s', type=str, help='Output directory')
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false')


    parser.set_defaults(use_cuda=True)
    args = parser.parse_args()
    save_dir = args.save_dir
    use_cuda = args.use_cuda

    print(args.model_path)

    args = load_params(args)
    args.save_dir = save_dir # overwrite from loaded params
    args.use_cuda = use_cuda # overwrite from loaded params

    main(args)
