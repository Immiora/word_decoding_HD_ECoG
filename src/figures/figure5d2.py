'''
PLot confusion matrix for speaker decoding

python figures/figure5d2.py \
    --task jip_janneke \
    --clf_type logreg \
    --res_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures \


'''
import sys
sys.path.insert(0, '.')

import os
import os.path as op
import argparse
import pandas as pd
from utils.plots import get_model_cmap
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def main(args):


    for imod, model in enumerate(args.model):
        mat = pd.read_csv(op.join(args.res_dir,  args.task, 'eval_' + model + '_classify_speakers_' + args.clf_type + '.csv'), index_col=0, header=0)
        mat = mat[mat['input']=='reconstructed']
        counts = pd.DataFrame(index= mat['target_label'].unique(), columns= mat['target_label'].unique())
        for t in mat['target_label'].unique():
            for p in mat['target_label'].unique():
                counts.loc[t, p] = mat[(mat['target_label']==t) & (mat['predicted_label']==p)].shape[0]

        d = counts.values.astype('float') * (100/12)
        plt.imshow(d, cmap=get_model_cmap(model), aspect='auto')
        plt.colorbar()
        plt.xlabel('All speakers')
        plt.ylabel('Targets (folds)')
        plt.yticks(range(len(counts)), mat['target_label'].unique())
        plt.xticks(range(len(counts)), mat['target_label'].unique(), rotation=45)


        if args.plot_dir != '':
            plt.savefig(op.join(args.plot_dir, 'fig5d_classify_speakers_' + model + '_' + args.clf_type + '_confusion_matrix.pdf'),
                        dpi=160, transparent=True)
            plt.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--model', '-m', type=str,  nargs="+",
                        choices=['mlp', 'densenet', 'seq2seq'],
                        default=['mlp', 'densenet', 'seq2seq'],
                        help='Model to run')
    parser.add_argument('--clf_type', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='logreg',
                        help='Type of classifier')
    parser.add_argument('--res_dir', '-o', type=str, help='Output directory', default='')
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
    args = parser.parse_args()

    main(args)