'''
Make spreadsheet for the behavioral experiment with Gorilla (.csv file)

python beh_exp/make_gorilla_spreadsheet_word_identify.py \
    --path_csv /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/spreadsheet_task1.csv \
    --path_stim /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_stimuli

Changes 2/12/2021:
    - only opt-true
    - relax constraint on same-syllable for alternative answer option
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
    d = pd.DataFrame({'word': ['boe', 'ga', 'waar', 'ik', 'jip', 'hoed', 'kom',
                                     'plukken', 'spelen',
                                     'grootmoeder', 'allemaal', 'janneke'],
                            'syl': [1, 1, 1, 1, 1, 1, 1,
                                    2, 2,
                                    3, 3, 3]})

    # load spreadsheet
    data = pd.read_csv(args.path_csv, sep=';')

    # clear contents
    data = data.drop(data[data['display'] != ''].index)

    # load filenames of audio stimuli
    wavs = [op.basename(i) for i in glob.glob(op.join(args.path_stim, '*.wav'))]

    # take only names of reconstructions, target names are the same
    # wavs_recon = [i for i in wavs if '_recon' in i]
    wavs_recon = [i for i in wavs if 'opt-true' in i]
    wavs_recon = [i.replace('.wav', '.mp3') for i in wavs_recon]

    # randomize order of stimuli
    random.shuffle(wavs_recon)

    # extract correct word based on filenames
    ans = [i.split('word-')[1].split('_')[0] for i in wavs_recon]

    # take random word of same number of syllables as second option
    # res2_ = [d[d['syl']==d[d['word']==i]['syl'].values[0]]['word'].drop(d[d['word'] == i].index).sample().values[0]
    #                                                                                                     for i in ans]
    res2_ = [d.drop(d[d['word'] == i].index).sample()['word'].values[0] for i in ans]

    # randomize place of correct word between res1 and res2 columns
    res1, res2 = zip(*[random.sample(sublist, 2) for sublist in zip(ans, res2_)])

    # put all info together
    block = {'display': ['Q1 audio - text'] * len(wavs_recon),
             'reconstructions_1': wavs_recon,
             'ans_1': ans,
             'res_1.1': res1,
             'res_1.2': res2}

    # add to the spreadsheet
    data = data.append(pd.DataFrame(block))

    # only keep targets of one model (save time)
    ix = data[(data['reconstructions_1'].str.contains('target')) & (~data['reconstructions_1'].str.contains('mod-seq2seq'))].index
    data = data.drop(ix).reset_index(drop=True)

    # set randomise_trials to 1
    data['randomise_trials'] = 1

    # add start, instrucitons, end trials
    # data = pd.concat([data, pd.DataFrame({'display': ['End']})])
    # data = pd.concat([pd.DataFrame({'display': ['Instructions text']}), data])
    data = pd.concat([pd.DataFrame({'display': ['Instructions text'], 'reconstructions_1': ['beh_word_id.png']}), data])

    # save spreadsheet
    data.to_csv(args.path_csv.replace('.csv', '_full.csv'), na_rep='', sep=',', index=False)


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare Gorilla spreadsheet')
    parser.add_argument('--path_csv',  type=str, help='Path to the spreadsheet')
    parser.add_argument('--path_stim', type=str, help='Path to the stimuli folder')
    args = parser.parse_args()
    main(args)
