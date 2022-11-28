'''
Evaluation script:

    - pick up the model and parameters
    - generate necessary info per cross-validation fold: fragments with 1 word per fold
    - train model per fold

Make a bash script to run
    - this: cross-val test per word
    - switch conda env
    - synthesize test sound per fold (generate_sound_from_decoder_v4)
    - switch conda env
    - run evaluation per fold


python evaluate_decoders/optuna_test_per_fold.py \
    --model_path path_to_optuna_$task_$subject_$model_$trial_$model_path \
    --k_fold 0 \
    --save_dir  path_to_optuna_$task_$subject_$model_$trial_eval

Update August 27:
    get_fragments from context_windows_jip_janneke_v2
    fixed unequal number of all fragments depending on the size of the context: .360 used to have more windows because only
    last point was required to be speech, .160 had less fragments. Now fragments are centered on the speech onsets and offsets,
    context agjusted around them -> same number of fragments regardless of the context size

'''

import sys
sys.path.insert(0, '.')

import os.path as op
import pandas as pd
import argparse
from os import makedirs
from utils.arguments import load_params
from train_decoders.optuna_decode_all_hfb_mel import trainer
from prepare_data.context_windows_jip_janneke_v2 import get_fragments

def run_cross_fold(args):

    # get fragments file per fold
    if not op.exists(args.save_dir): makedirs(args.save_dir)
    word_fragments = pd.read_csv(args.subsets_path, index_col=None,  header=0)
    ind = word_fragments[word_fragments['subset']=='test'].index[args.k_fold]
    word_fragments['subset'] = 'train'
    word_fragments.loc[ind, 'subset'] = 'validation' # training script uses train & validation
    #temp_dir = tempfile.TemporaryDirectory()
    temp_subsets = op.join(args.save_dir,
                           op.basename(args.subsets_path).replace('.csv', '_fold' + str(args.k_fold) + '.csv'))
    word_fragments.to_csv(temp_subsets)
    step = float(args.fragments.split('step')[1].split('s_')[0])
    context = float(args.fragments.split('window')[1].split('s_')[0])
    full_fragments, speech_fragments = get_fragments(temp_subsets, step, context)
    temp_fragments = op.join(args.save_dir, 'fragments_fold' + str(args.k_fold) + '_speech_only.csv')
    speech_fragments.to_csv(temp_fragments)
    full_fragments.to_csv(temp_fragments.replace('speech_only', 'full'))

    args.fragments = temp_fragments
    args.fragments_full = temp_fragments.replace('speech_only', 'full')
    args.output_mean = args.output_mean.replace('.npy', '_eval_fold' + str(args.k_fold) + '.npy')
    args.output_std = args.output_std.replace('.npy', '_eval_fold' + str(args.k_fold) + '.npy')
    args.subsets_path_fold = temp_subsets # test this

    # train
    best_val_loss = trainer(args)


## load normalization parameters
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--model_path', '-m', type=str, help='Directory with optuna model, whose parameters will be used')
    parser.add_argument('--k_fold', '-k', type=int, help='Index of the test word')
    # parser.add_argument('--subsets_path', type=str, help='Subset file')
    parser.add_argument('--save_dir', '-s', type=str, help='Output directory', default='')
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false')
    parser.set_defaults(use_cuda=True)
    args = parser.parse_args()
    save_dir = op.join(args.save_dir, 'fold' + str(args.k_fold))
    use_cuda = args.use_cuda

    args = load_params(args)
    args.optuna_model = args.model_path
    delattr(args, 'model_path')
    args.save_dir = save_dir # overwrite from loaded params
    args.use_cuda = use_cuda # overwrite from loaded params

    run_cross_fold(args)
