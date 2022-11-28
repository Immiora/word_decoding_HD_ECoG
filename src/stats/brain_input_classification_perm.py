'''
Classify words based on brain input



nice python stats/brain_input_classification_perm.py \
    --task jip_janneke \
    --subject subject1 \
    --clf_type logreg \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/brain_input_decoding \
    --n_perms 120



'''

import sys
sys.path.insert(0, '.')

from numpy import random
import os
import argparse
import pandas as pd
import os.path as op
import glob

from decimal import Decimal, ROUND_HALF_UP
from evaluate_decoders.eval_word_classification import classify_words

from utils.datasets import BrainDataset, split_data, load_data, get_moments
from utils.general import make_save_dir
from utils.plots import plot_eval_metric_mean, plot_eval_metric_box
from utils.private.datasets import get_annot_by_code

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sec2ind = lambda s, sr: int(Decimal(s * sr).quantize(0, ROUND_HALF_UP))

def main(args):

    # select parameters
    params = {}
    params['sr'] = [15, 25, 75]
    params['input_band'] = ['70-170', '60-300']
    params['input_ref'] = ['car', 'bipolar']

    word_annotations = get_annot_by_code(args.subject)
    dir_name = op.join('/Fridge/users/julia/project_decoding_' + args.task, 'data', args.subject)

    for clf in args.clf_type:
        args.model_type = clf + '_word_id_perm'
        tmp = os.path.join(args.save_dir, args.task, args.subject, args.model_type)

        for sr in params['sr']:

            for band in params['input_band']:

                for ref in params['input_ref']:

                    for clf_feat in args.clf_feat:
                        save_dir = os.path.join(tmp, clf_feat + '_sr' + str(sr) + \
                                                     '_' + str(ref) + \
                                                     '_hfb' + str(band) + \
                                                     '_flen' + str(1))
                        save_dir = make_save_dir(save_dir)
                        plotdir = save_dir.replace('results', 'pics/decoding')
                        if not os.path.isdir(plotdir): os.makedirs(plotdir)

                        input_file = op.join(dir_name,
                                                 args.subject + '_hfb_' + \
                                                 args.task + '_' + \
                                                 ref + '_' + \
                                                 band + '_' + \
                                                 str(sr) + 'Hz.npy')
                        input_mean_file = input_file.replace('.npy', '_rest_precomputed_mean.npy')
                        input_std_file = input_file.replace('.npy', '_rest_precomputed_std.npy')

                        # get same size window across words + necessary context: contexts_full
                        # set up loop over all words
                        # word_fragments: entry per word, not contexts, they are irrelevant here
                        word_fragments = pd.read_csv(op.join(dir_name, word_annotations[args.subject]), index_col=None, header=0)
                        results = pd.DataFrame()

                        for perm in range(args.n_perms):
                            fold = random.randint(0, word_fragments.shape[0])
                            word_fragments['subset'] = 'train'
                            word_fragments.loc[fold, 'subset'] = 'validation'  # training script uses train & validation

                            # shaffle word labels
                            word_fragments['text'] = word_fragments['text'].values[random.permutation(word_fragments.shape[0])]

                            OFFSET = 0 # 0
                            LEN = 1 # 1

                            # seconds in word_fragments to samples in args.sr
                            fragments = word_fragments[['xmin']].copy() - OFFSET
                            fragments['subset'] = word_fragments['subset']

                            # save temp fragments_for_classif
                            fragments.to_csv(args.subject + '_temp.csv')

                            # get data
                            dataset = BrainDataset(input_file, input_file, args.subject + '_temp.csv',
                                                  sr, sr, LEN, brain_delay=0)
                            trainset, valset, testset = split_data(dataset)
                            x_mean, x_std, _, _, _ = get_moments(trainset, input_mean_file, input_std_file,
                                                                            input_mean_file, input_std_file, use_pca=False,
                                                                            n_pcs=100, can_write=False)

                            # classify same time points: train on brain data, test on brain data
                            train_loader, val_loader, _ = load_data(trainset, valset, testset,
                                                                              batch_size=len(trainset),
                                                                              shuffle_train=False)
                            train_input, _, _ = next(iter(train_loader))
                            val_input, _, _ = next(iter(val_loader))

                            res_clf_input = classify_words(word_fragments, train_input, val_input, val_input,
                                                           x_mean, x_std,
                                                           input_type=clf_feat,
                                                           clf_type=clf)
                            res_clf_input = res_clf_input[1:]
                            res_clf_input['input'] = 'brain_input'

                            results = results.append(res_clf_input.iloc[-1])
                        print(save_dir)
                        print(results['accuracy'].mean())

                        # make plots: word classification
                        #plot_eval_metric_mean('classify', pd.DataFrame([results['accuracy'].mean()]), plotdir=None)
                        plot_eval_metric_mean('classify', pd.DataFrame([results['accuracy'].mean()]), plotdir=plotdir)

                        # save
                        results.to_csv(os.path.join(save_dir, 'perm_classify.csv'))

        results = pd.DataFrame()
        plotdir = tmp.replace('results', 'pics/decoding')
        if not os.path.isdir(plotdir): os.makedirs(plotdir)
        for clf_feat in args.clf_feat:
            all_dirs = glob.glob(op.join(tmp, clf_feat + '_*'))
            assert len(all_dirs) == 12, 'number of dirs is ' + str(len(all_dirs))
            for d in all_dirs:
                a = pd.read_csv(op.join(d, 'perm_classify.csv'), index_col=[0])
                b = {}
                b['accuracy'] = a['accuracy'].mean()
                b['model'] = [clf_feat]
                b['dir'] = d
                results = results.append(pd.DataFrame(b))

        plot_eval_metric_box('classify', pd.DataFrame(results), by='model', plotdir=plotdir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str, choices=['jip_janneke', 'hp_reading'])
    parser.add_argument('--subject', '-s', type=str)
    parser.add_argument('--clf_type', '-c', type=str,
                        choices=['svm_linear', 'svm_rbf', 'logreg', 'mlp'], default='svm_linear',
                        help='Type of classifier')
    parser.add_argument('--clf_feat', '-f', type=str,  nargs="+",
                        choices=['default', 'avg_time', 'avg_chan'],
                        default=['default', 'avg_time', 'avg_chan'],
                        help='Input classifier features')
    parser.add_argument('--n_perms', '-n', type=int, help='Number of permutations', default=1000)
    parser.add_argument('--save_dir', '-o', type=str, help='Output directory')

    parser.set_defaults(use_cuda=True)
    args = parser.parse_args()

    main(args)


## check brain plots for subjects with bad performance

# run before classify_words(word_fragments, train_input, val_input, val_input, x_mean, x_std)

# import matplotlib.pyplot as plt
# import numpy as np
# from utils.training import preprocess_t
#
# ##
# x = np.load(args.input_file)
# ##
# plt.figure()
# plt.imshow(x[15:15*30].T, aspect='auto')
#
# ##
# xz = (x - x_mean.detach().numpy()[None,:]) / x_std.detach().numpy()[None,:]
#
# ##
# plt.figure()
# plt.imshow(xz[:15*30].T, aspect='auto')
#
# ##
# x_train = preprocess_t(train_input, x_mean, x_std, clip_t_value=None, DEVICE='cpu').detach().numpy()
#
# ##
# train_input = train_input.detach().numpy()
#
#
# ##
# plt.figure()
# plt.imshow(train_input[0].T, aspect='auto')
#
# ##
# labels = pd.factorize(word_fragments['text'])[0]
# val_labels = labels[word_fragments[word_fragments['subset'] == 'validation'].index]
# train_labels = np.delete(labels, word_fragments[word_fragments['subset'] == 'validation'].index)
#
# # for lab in np.unique(train_labels):
# #     plt.figure()
# #     plt.imshow(np.mean(train_input[train_labels == lab], 0).T, aspect='auto')
# #     plt.title(word_fragments.loc[labels==lab]['text'].values[0])
#
# ##
# plt.subplots(3, 4)
# for lab in np.unique(train_labels):
#     plt.subplot(3, 4, lab+1)
#     plt.imshow(np.mean(x_train[train_labels == lab], 0), aspect='auto')
#     plt.title(word_fragments.loc[labels==lab]['text'].values[0])
#
# ## temporal profiles only
# plt.subplots(3, 4)
# for lab in np.unique(train_labels):
#     plt.subplot(3, 4, lab+1)
#     plt.imshow(np.mean(np.mean(x_train[train_labels == lab], 0), 0)[:, None], aspect='auto')
#     plt.title(word_fragments.loc[labels==lab]['text'].values[0])
#
# ## spectral profiles only
# plt.subplots(3, 4)
# for lab in np.unique(train_labels):
#     plt.subplot(3, 4, lab+1)
#     plt.imshow(np.mean(np.mean(x_train[train_labels == lab], 0), -1)[:, None], aspect='auto')
#     plt.title(word_fragments.loc[labels==lab]['text'].values[0])
