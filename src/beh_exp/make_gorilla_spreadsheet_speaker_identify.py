'''
Make spreadsheet for the behavioral experiment with Gorilla (.csv file)

python beh_exp/make_gorilla_spreadsheet_speaker_identify.py \
    --path_csv /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/spreadsheet_task2.csv \
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
    d = pd.DataFrame({'id': list(range(1, 6)),
                            'sex': ['f', 'f', 'f', 'm', 'm']})

    # load spreadsheet
    data = pd.read_csv(args.path_csv, sep=';')

    # clear contents
    data = data.drop(data[data['display'] != ''].index)

    # load filenames of audio stimuli
    wavs = [op.basename(i) for i in glob.glob(op.join(args.path_stim, '*.wav'))]

    # take only names of reconstructions, target names are the same
    # wavs_recon = [i for i in wavs if '_recon' in i]
    wavs_recon = [i for i in wavs if 'opt-true_recon' in i]

    # randomize order of stimuli
    random.shuffle(wavs_recon)

    # extract correct word based on filenames
    ans = [i.replace('_recon', '_target') for i in wavs_recon]

    # take random subject of same sex
    sub = lambda x: int(x.split('sub-')[1].split('_')[0])
    rep = lambda x, y: x.replace('sub-' + x.split('sub-')[1].split('_')[0], 'sub-' + str(y))
    # res2_ = [rep(i, d[d['sex']==d[d['id']==sub(i)]['sex'].values[0]]['id'].drop(d[d['id'] == sub(i)].index).sample().values[0])
    #                                                                                                    for i in ans]
    res2_ = [rep(i, d.drop(d[d['id'] == sub(i)].index).sample()['id'].values[0]) for i in ans]


    # randomize place of correct response between res1 and res2 columns
    res1, res2 = zip(*[random.sample(sublist, 2) for sublist in zip(ans, res2_)])

    # put all info together
    block = {'display': ['Q2 audio - audio'] * len(wavs_recon),
             'reconstructions_2': wavs_recon,
             'ans_2': ans,
             'res_2.1': res1,
             'res_2.2': res2}

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
