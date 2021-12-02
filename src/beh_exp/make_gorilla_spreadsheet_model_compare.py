'''
Make spreadsheet for the behavioral experiment with Gorilla (.csv file)

python beh_exp/make_gorilla_spreadsheet_model_compare.py \
    --path_csv /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/spreadsheet_task3.csv \
    --path_stim /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_stimuli

'''

import argparse
import os.path as op
import pandas as pd
import os
import glob
import random

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(args):

    # set options: different syllables across words
    d = pd.DataFrame({'m': ['densenet', 'mlp', 'seq2seq']})

    # load spreadsheet
    data = pd.read_csv(args.path_csv, sep=';')

    # clear contents
    data = data.drop(data[data['display'] != ''].index)

    # load filenames of audio stimuli
    wavs = [os.path.basename(i) for i in glob.glob(op.join(args.path_stim, '*.wav'))]

    # take only names of reconstructions, target names are the same
    wavs_recon = [i for i in wavs if '_recon' in i]

    # randomize order of stimuli
    wavs_opt = [i for i in wavs_recon if 'opt-true' in i]
    wavs_opt = random.sample(wavs_opt, len(wavs_opt))
    ans = wavs_opt.copy()

    # extract correct word based on filenames
    words = [i.split('word-')[1].split('_')[0] for i in wavs_opt]

    # once for opt vs non-opt, another run for densenet/mlp/seq2seq
    for run in range(2):

        # take random word of same number of syllables as second option
        if run == 0:
            # alternative response: non-optimized
            res2_ = [i.replace('opt-true', 'opt-false') for i in wavs_opt]
        else:
            # alternative response: one of the other two models (densenet/mlp/seq2seq)
            sub = lambda x: x.split('mod-')[1].split('_')[0]
            rep = lambda x, y: x.replace('mod-' + x.split('mod-')[1].split('_')[0], 'mod-' + y)
            res2_ = [rep(i, d.drop(d[d['m'] == sub(i)].index).sample().values.flatten()[0])  for i in wavs_opt]

        # randomize place of correct word between res1 and res2 columns
        res1, res2 = zip(*[random.sample(sublist, 2) for sublist in zip(ans, res2_)])

        # put all info together
        block = {'display': ['Q3 evaluations'] * len(wavs_opt),
                 'reconstructions_3': wavs_opt,
                 'ans_3': ans,
                 'word_3': words,
                 'res_3.1': res1,
                 'res_3.2': res2}

        # add to the spreadsheet
        data = data.append(pd.DataFrame(block))

    # set randomise_trials to 1
    data['randomise_trials'] = 1
    # data['color_background_1'] = 'Yellow.001.jpeg'
    # data['color_background_2'] = 'Blue.001.jpeg'

    # add End trial to the end
    data = data.append(pd.DataFrame({'display': ['End']}))

    # save spreadsheet
    data.to_csv(args.path_csv.replace('.csv', '_full.csv'), na_rep='', sep=',', index=False)


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare Gorilla spreadsheet')
    parser.add_argument('--path_csv',  type=str, help='Path to the spreadsheet')
    parser.add_argument('--path_stim', type=str, help='Path to the stimuli folder')
    args = parser.parse_args()
    main(args)

##
# z[z['reconstructions_1'].str.contains('sub-1') & z['reconstructions_1'].str.contains('mod-seq2seq') & z['reconstructions_1'].str.contains('opt-true')]['Correct'].mean()