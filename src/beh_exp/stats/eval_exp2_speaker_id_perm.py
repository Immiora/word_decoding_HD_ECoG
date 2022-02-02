'''
Run permutation test on behavioral data from exp 2

python beh_exp/stats/eval_exp2_speaker_id_perm.py \
    --data_dir /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/subjects/exp2_29subjs \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_exp/exp2_29subjs \
    --n_perms 1000
'''

import sys
sys.path.insert(0, '.')
import argparse

from beh_exp.stats.eval_exp1_word_id_perm import run_main



##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process exp 1 or 2')
    parser.add_argument('--data_dir',  type=str, help='Path to the dir with spreadsheets for exp 1 or 2')
    parser.add_argument('--save_dir', type=str, help='Path to results')
    parser.add_argument('--n_perms', '-n', type=int, help='Number of permutations', default=1000)
    args = parser.parse_args()
    args.type_exp = 'exp2'
    run_main(args)
