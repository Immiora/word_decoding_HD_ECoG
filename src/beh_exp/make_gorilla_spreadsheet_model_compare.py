'''
Make spreadsheet for the behavioral experiment with Gorilla (.csv file)

python beh_exp/make_gorilla_spreadsheet_model_compare.py \
    --path_csv /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/spreadsheet_task3.csv \
    --path_stim /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_stimuli

Changes 24/12/2021:
    - Only run opt/non-opt, and not the model comparisons, there are more difficult,and probably the effects are too small

Changes 15/12/2021:
    - Dutch words for options

'''

import argparse
import os.path as op
import pandas as pd
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
    wavs = [op.basename(i) for i in glob.glob(op.join(args.path_stim, '*.wav'))]

    # take only names of reconstructions, target names are the same
    wavs_recon = [i for i in wavs if '_recon' in i]
    wavs_recon = [i.replace('.wav', '.mp3') for i in wavs_recon]

    # randomize order of stimuli
    wavs_opt = [i for i in wavs_recon if 'opt-true' in i]
    wavs_opt = random.sample(wavs_opt, len(wavs_opt))
    tar1 = wavs_opt.copy()

    # extract correct word based on filenames
    words = [i.split('word-')[1].split('_')[0] for i in wavs_opt]
    words= ['<p style="font-size: 300%;">' + i + '</p>' for i in words]

    # once for opt vs non-opt, another run for densenet/mlp/seq2seq
    # for run in range(2):
    for run in range(1):

        # take random word of same number of syllables as second option
        if run == 0:
            # alternative response: non-optimized
            tar2_ = [i.replace('opt-true', 'opt-false') for i in wavs_opt]
        else:
            # alternative response: one of the other two models (densenet/mlp/seq2seq)
            sub = lambda x: x.split('mod-')[1].split('_')[0]
            rep = lambda x, y: x.replace('mod-' + x.split('mod-')[1].split('_')[0], 'mod-' + y)
            tar2_ = [rep(i, d.drop(d[d['m'] == sub(i)].index).sample().values.flatten()[0])  for i in wavs_opt]

        # randomize place of correct word between res1 and res2 columns
        tar1, tar2 = zip(*[random.sample(sublist, 2) for sublist in zip(tar1, tar2_)])
        res1 = ['Optie 1'] * len(tar1)
        res2 = ['Optie 2'] * len(tar1)

        # add targets to make the task a bit easier and nicer (maybe could use them for ceilings?)
        tar1_catch = random.sample(wavs_opt.copy(), 60)
        tar2_catch_ = [i.replace('recon', 'target') for i in tar1_catch]
        tar1_catch, tar2_catch = zip(*[random.sample(sublist, 2) for sublist in zip(tar1_catch, tar2_catch_)])
        res1_catch = ['Optie 1'] * len(tar1_catch)
        res2_catch = ['Optie 2'] * len(tar1_catch)
        words_catch = [i.split('word-')[1].split('_')[0] for i in tar1_catch]
        words_catch = ['<p style="font-size: 300%;">' + i + '</p>' for i in words_catch]

        # put all info together
        # block = {'display': ['Q3 evaluations'] * len(wavs_opt),
        #          #'ans_3': ans,
        #          'word_3': words,
        #          'reconstructions_3.1': tar1,
        #          'reconstructions_3.2': tar2,
        #          'res_3.1': res1,
        #          'res_3.2': res2}
        block = {'display': ['Q3 evaluations'] * (len(wavs_opt) + len(tar1_catch)),
                 #'ans_3': ans,
                 'word_3': words + words_catch,
                 'reconstructions_3.1': tar1 + tar1_catch,
                 'reconstructions_3.2': tar2 + tar2_catch,
                 'res_3.1': res1 + res1_catch,
                 'res_3.2': res2 + res2_catch}

        # add to the spreadsheet
        data = data.append(pd.DataFrame(block))

    # set randomise_trials to 1
    data['randomise_trials'] = 1

    # add start, instrucitons, end trials
    # data = pd.concat([data, pd.DataFrame({'display': ['End']})])
    # data = pd.concat([pd.DataFrame({'display': ['Instructions evaluations']}), data]) # add test audio for Start
    data = pd.concat([pd.DataFrame({'display': ['Instructions evaluations'], 'reconstructions_3.1': ['beh_model_compare.png']}), data])

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