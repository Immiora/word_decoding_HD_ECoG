'''

Run permutation testing: shift brain data before running through the pretrained network

Originally saved predictions and targets (.npy) after denoralization: scaling and re-centering output of the model pass,
same was applied to the targets. That operation alone produces very high correlations, and results in a high baseline for
permuted (shifted) data (eval_pearsonr_vad_stoi_perm_shifts).

Calculate correlation without using denormalization, just normalized targets and output from the model

First tried shift brain data by a random int samples in [5, 30] seconds forward or back
# args.input_delay = np.random.randint(args.input_sr * 5, args.input_sr * 30) * (1 if np.random.random() < 0.5 else -1)

But this results in all kinds of speech+silence mix, and very wide distributions for permutations
Instead do only non-speech input here, and separately calculate pearsonr and stoi on permuted words

1. Run synth_speech_targets_reconstructed, pass modified args:
    - change brain shift by a random int forward or backward
    - change save_dir for synthesized audios -> temo_perm under fold?
    This is for calculating VAD perm and STOI perm
2. Make a random brain shift here, calculate pearsonr
3. Save results per metric

because need to synth audios run within parallel_wavegan conda environment

python stats/eval_pearsonr_vad_stoi_perm_shifts.py \
    --task jip_janneke \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
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
import tempfile

from models.select_model import select_model
from utils.training import *
from utils.arguments import load_params
from utils.datasets import BrainDataset, split_data, load_data, get_moments
from utils.plots import plot_eval_metric_box
from utils.general import sec2ind
from utils.private.datasets import get_subjects_by_code

from evaluate_decoders.eval_pearsonr import calculate_pearsonr_flattened
from evaluate_decoders.eval_vad import calculate_vad
from evaluate_decoders.eval_stoi import calculate_stoi
from synthesize_audio.synth_speech_targets_reconstructed import synthesize


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

            predictions.append(out.cpu().detach().numpy().squeeze())
            targets.append(preprocess_t(t, t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy().squeeze())
        return np.array(predictions), np.array(targets)


    for isubj, subject in enumerate(args.subject):
        for imod, model in enumerate(args.model):
            gen_dir = op.join(args.res_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))['_number']
            trial_dir = op.join(gen_dir, str(best_trial), 'eval')

            res_pearsonr_perm = pd.DataFrame()
            res_vad_perm = pd.DataFrame()
            res_stoi_perm = pd.DataFrame()

            tmp_perm_dir = tempfile.mkdtemp()

            for perm in range(args.n_perms):

                # select a test fold (validation word) at random
                print(perm)
                res_pearsonr, res_vad, res_stoi = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

                for fold in range(12):
                    fold_dir = op.join(trial_dir, 'fold' + str(fold))
                    args.model_path = op.join(fold_dir,
                                              op.basename([f.path for f in os.scandir(op.join(gen_dir, str(best_trial))) if
                                                           f.is_dir() and 'eval' not in f.path][0]))
                    args = load_params(args)
                    word_len_sampl = args.input_sr

                    # get correct filename for fragments_for_classif = from the fold number + subsets_full_for_classif
                    fragments_for_classif_fold = op.join(fold_dir, 'fragments_fold'+str(fold)+'_full_for_classif.csv')
                    word_fragments_fold = op.join(fold_dir, op.basename(args.subsets_path).replace('.csv', '_fold'+str(fold)+'.csv'))
                    word_fragments = pd.read_csv(word_fragments_fold, index_col=0, header=0)
                    sil_fragments = pd.DataFrame({'xmin': [0] + list(word_fragments['xmax'].values[:-1] + .5),
                                                  'xmax': word_fragments['xmin'].values - .5})

                    # get random index for sil_fragment, random float within + duration < offset of silence
                    rand_ix = sil_fragments.sample().index.values[0]
                    target_dur = word_fragments[word_fragments['subset']=='validation']['duration'].values[0]

                    # make sure sil_fragments.loc[rand_ix, 'xmax'] - sil_fragments.loc[rand_ix, 'xmin'] is large enough (>=duration) !!!
                    while sil_fragments.loc[rand_ix, 'xmax'] - sil_fragments.loc[rand_ix, 'xmin'] < target_dur:
                        rand_ix = sil_fragments.sample().index.values[0]
                    sil_onset = np.random.uniform(sil_fragments.loc[rand_ix, 'xmin'], sil_fragments.loc[rand_ix, 'xmax'] - target_dur)

                    # input delay by difference between onset of validation in word_fragments and found index of silence
                    args.input_delay = sec2ind(sil_onset - word_fragments[word_fragments['subset']=='validation']['xmin'].values[0], args.input_sr)

                    if 'pearsonr' in args.metric:
                        # load data for model pass: 1-sec long, consistent with matched data and word classification metric
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
                        val_predictions, val_targets = model_pass(val_loader)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        # pearson correlation
                        print('Computing pearsonr')
                        temp = calculate_pearsonr_flattened(val_predictions, val_targets) # awful format!!!
                        res_pearsonr = res_pearsonr.append(temp, ignore_index=True)

                    if 'vad' in args.metric or 'stoi' in args.metric:
                        # call to synth_speech_targets here to generate permuted audios
                        args.save_dir = tmp_perm_dir
                        args.fragments = op.join(fold_dir, 'fragments_fold'+str(fold)+'_speech_only.csv')
                        args.checkpoint = 'checkpoint_499'
                        args.target_set = 'validation'
                        args.synth_targets = True
                        synthesize(args) # check save_path

                        # grab paths to generated perm audio
                        args.audio_val_targets = os.path.join(args.save_dir, 'parallel_wavegan', 'targets_validation_gen.wav')
                        args.audio_val_predictions = os.path.join(args.save_dir, 'parallel_wavegan',
                                                                 args.model_type + '_validation_gen.wav')
                        #args.audio_val_targets = '/tmp/tmpdhba4kxr/parallel_wavegan/targets_validation_gen.wav'
                        #args.audio_val_predictions = '/tmp/tmpdhba4kxr/parallel_wavegan/seq2seq_validation_gen.wav'
                        assert os.path.exists(args.audio_val_targets), 'File does not exist: ' + args.audio_val_targets
                        assert os.path.exists(args.audio_val_predictions), 'File does not exist: ' + args.audio_val_predictions

                        # VAD
                        if 'vad' in args.metric:
                            print('Computing VAD match')
                            temp = calculate_vad(args.audio_val_predictions, args.audio_val_targets)
                            res_vad = res_vad.append(temp, ignore_index=True)

                        # STOI
                        if 'stoi' in args.metric:
                            print('Computing STOI/ESTOI')
                            temp = calculate_stoi(args.audio_val_predictions, args.audio_val_targets)
                            res_stoi = res_stoi.append(temp, ignore_index=True)

                for metric in args.metric:
                    if metric == 'pearsonr':
                        res = {'pearsonr': [res_pearsonr['r'].mean()], 'perm': [perm]}
                        res_pearsonr_perm = res_pearsonr_perm.append(pd.DataFrame(res), ignore_index=True)
                    elif metric == 'vad':
                        res = {'vad_match': [res_vad['vad_match'].mean()], 'perm': [perm]}
                        res_vad_perm = res_vad_perm.append(pd.DataFrame(res), ignore_index=True)
                    elif metric == 'stoi':
                        res = {'stoi': [res_stoi['stoi'].mean()], 'estoi':res_stoi['estoi'].mean(), 'perm': [perm]}
                        res_stoi_perm = res_stoi_perm.append(pd.DataFrame(res), ignore_index=True)
                    else:
                        raise ValueError

            plotdir = trial_dir.replace('results', 'pics/decoding')
            for metric in args.metric:
                if metric == 'pearsonr':
                    res = res_pearsonr_perm
                elif metric == 'vad':
                    res = res_vad_perm
                elif metric == 'stoi':
                    res = res_stoi_perm
                else:
                    raise ValueError
                res.to_csv(os.path.join(trial_dir, 'eval_n_folds12_'+ metric+'_perm_sil_only.csv'))
                plot_eval_metric_box(metric, res, plotdir=plotdir)



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
    parser.add_argument('--metric', '-e', type=str,  nargs="+",
                        choices=['pearsonr', 'vad', 'stoi'],
                        default=['pearsonr', 'vad', 'stoi'],
                        help='Metric to use')
    parser.add_argument('--n_perms', '-n', type=int, help='Number of permutations', default=1000)
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)

    main(args)
