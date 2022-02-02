'''
Process behvaioral data: experiment 2 (speaker id)

Read in directory with all csv spreadsheets (one per subject)
Process each, populate final csv with results
Save csv with results to plot late in Figure 6a

python beh_exp/process/process_exp2_speaker_id.py \
    --data_dir /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/subjects/exp2_29subjs \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp2_29subjs
'''

import sys
sys.path.insert(0, '.')
import argparse

from beh_exp.process.process_exp1_word_id import run_main



##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process exp 2')
    parser.add_argument('--data_dir',  type=str, help='Path to the dir with spreadsheets for exp 2')
    parser.add_argument('--save_dir', type=str, help='Path to results')
    args = parser.parse_args()
    args.type_exp = 'exp2'
    run_main(args)
