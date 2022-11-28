'''
script to prepare stimuli for the behavioral experiment
subjext x word x type_model x optimized/not x target/recon

for tyoe_model: mlp, densenet, seq2seq
optimized/not: trial 0 of optuna vs optimized trial

The script does the following:
    1. find all generated audios (parallel wavegan)
    2. look up word label per test example
    3. save to output dir, add all info to the name of the file

Externally: apply pitch change, export to mp3 in Audacity (use macros for batch processing)

python beh_exp/make_stimuli.py \
    --task jip_janneke \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --n_folds 12
'''

import argparse
import os.path as op
import json
import pandas as pd
import os
import glob
from shutil import copyfile
from collections import OrderedDict
from utils.private.datasets import get_subjects_by_code

def main(args):
    subjects_list =args.subject
    model_list = ['mlp', 'densenet', 'seq2seq']
    output_dir = op.join(args.save_dir, args.task, 'beh_stimuli')
    if not op.isdir(output_dir): os.makedirs(output_dir)

    for subject in args.subject:
        print(subject)

        for model in args.model:
            gen_dir = op.join(args.save_dir, args.task, subject, model)
            best_trial = json.load(open(op.join(gen_dir, 'best_trial.json'), 'r'))

            for trial in [0, best_trial['_number']]: # optimized and non-optimized
                trial_dir = op.join(gen_dir, str(trial), 'eval')
                model_dir = op.basename([f.path for f in os.scandir(op.join(gen_dir, str(trial))) if f.is_dir() and 'eval' not in f.path][0])
                if trial != 0: print(model_dir)

                for fold in range(args.n_folds):
                    fold_dir = op.join(trial_dir, 'fold' + str(fold))
                    target_wav = None
                    recon_wav = None

                    # audio target and recon
                    #subdirs = [x[0] for x in os.walk(trial_dir)]
                    wav_dir = op.join(fold_dir, model_dir, 'parallel_wavegan')
                    assert op.exists(wav_dir), 'No wav dir at ' + wav_dir
                    wavs = glob.glob(op.join(wav_dir, '*.wav'))
                    assert len(wavs) == 2, 'Number of wav files:  ' + str(len(wavs))
                    for wav in wavs:
                        if 'targets' in wav:
                            target_wav = wav
                        else:
                            recon_wav = wav

                    # word label
                    subsets = pd.read_csv(glob.glob(op.join(fold_dir, '*_fold'+ str(fold) + '.csv'))[0])
                    label = subsets[subsets['subset']=='validation']['text'].values[0]

                    # copy and save audios
                    opt_tag = 'false' if trial == 0 else 'true'
                    new_name = 'sub-' + str(subjects_list.index(subject)+1) + '_word-' + label + '_mod-' + model + '_opt-' + opt_tag
                    copyfile(target_wav, op.join(output_dir, new_name + '_target.wav'))
                    copyfile(recon_wav, op.join(output_dir, new_name + '_recon.wav'))

    info = OrderedDict({'sub':{}, 'word':{}, 'mod':{}, 'opt':{}})
    for i in range(len(subjects_list)):
        info['sub'][i+1] = subjects_list[i]
    info['word'] = {1:'boe', 2:'hoed', 3:'grootmoeder', 4:'ga', 5:'waar', 6:'allemaal', 7:'jip',
         8:'plukken', 9:'spelen', 10:'ik', 11:'janneke', 12:'kom'}
    for i in range(len(model_list)):
        info['mod'][i+1] = model_list[i]
    info['opt'] = {'true':'best_trial', 'false': 'trial 0'}

    json.dump(info,
              open(op.join(output_dir, 'stimuli.json'), 'w'),
              indent=4,
              sort_keys=True)


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
    parser.add_argument('--save_dir', '-o', type=str, help='Output directory', default='')
    parser.add_argument('--n_folds', '-n', type=int, help='Number of folds')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)
    main(args)