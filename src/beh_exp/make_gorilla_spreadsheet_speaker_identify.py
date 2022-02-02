'''
Make spreadsheet for the behavioral experiment with Gorilla (.csv file)

python beh_exp/make_gorilla_spreadsheet_speaker_identify.py \
    --path_csv /Fridge/users/julia/project_decoding_jip_janneke/data/beh_exp/spreadsheet_task2.csv \
    --path_stim /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/beh_stimuli

Changes 15/12/2021:
    - Dutch words for options

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

    # set options: speakers of different sex
    d = pd.DataFrame({'id': list(range(1, 6)),
                            'sex': ['f', 'f', 'f', 'm', 'm']})

    # load spreadsheet
    data = pd.read_csv(args.path_csv, sep=';')

    # clear contents
    data = data.drop(data[data['display'] != ''].index)

    # load filenames of audio stimuli
    wavs = [op.basename(i) for i in glob.glob(op.join(args.path_stim, '*.wav'))]

    # take only names of reconstructions, target names are the same
    # wavs_recon = [i for i in wavs if '_recon' in i] # now use both targets and recon to have ceiling
    wavs_recon = [i for i in wavs if 'opt-true' in i]
    wavs_recon = [i.replace('.wav', '.mp3') for i in wavs_recon]

    # randomize order of stimuli
    random.shuffle(wavs_recon)

    # extract correct word based on filenames
    #ans = [i.replace('_recon', '_target') for i in wavs_recon]
    tar1 = [i.replace('_recon', '_target') for i in wavs_recon]
    ans = ['Spreker ' + i.split('sub-')[1].split('_')[0] for i in wavs_recon]

    # take random subject of same sex
    sub = lambda x: int(x.split('sub-')[1].split('_')[0])
    rep = lambda x, y: x.replace('sub-' + x.split('sub-')[1].split('_')[0], 'sub-' + str(y))
    # tar2_ = [rep(i, d[d['sex']==d[d['id']==sub(i)]['sex'].values[0]]['id'].drop(d[d['id'] == sub(i)].index).sample().values[0])
    #                                                                                                    for i in ans]
    tar2_ = [rep(i, d.drop(d[d['id'] == sub(i)].index).sample()['id'].values[0]) for i in tar1]


    # randomize place of correct response between res1 and res2 columns
    tar1, tar2 = zip(*[random.sample(sublist, 2) for sublist in zip(tar1, tar2_)])
    res1 = ['Spreker ' + i.split('sub-')[1].split('_')[0] for i in tar1]
    res2 = ['Spreker ' + i.split('sub-')[1].split('_')[0] for i in tar2]

    # put all info together
    block = {'display': ['Q2 audio - audio'] * len(wavs_recon),
             'reconstructions_2': wavs_recon,
             'ans_2': ans,
             'targets_2.1': tar1,
             'targets_2.2': tar2,
             'res_2.1': res1,
             'res_2.2': res2}

    # add to the spreadsheet
    data = data.append(pd.DataFrame(block))

    # only keep targets of one model (save time)
    ix = data[(data['reconstructions_2'].str.contains('target')) & (data['targets_2.1'].str.contains('target')) & (~data['targets_2.1'].str.contains('mod-seq2seq'))].index
    data = data.drop(ix).reset_index(drop=True)

    # set randomise_trials to 1
    data['randomise_trials'] = 1

    # add start, instrucitons, end trials
    # data = pd.concat([data, pd.DataFrame({'display': ['End']})])
    # data = pd.concat([pd.DataFrame({'display': ['Instructions audio']}), data]) # add test audio for Start
    data = pd.concat([pd.DataFrame({'display': ['Instructions audio'], 'reconstructions_2': ['beh_spreker_id.png']}), data])

    # save spreadsheet
    data.to_csv(args.path_csv.replace('.csv', '_full.csv'), na_rep='', sep=',', index=False)


##
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Prepare Gorilla spreadsheet')
    parser.add_argument('--path_csv',  type=str, help='Path to the spreadsheet')
    parser.add_argument('--path_stim', type=str, help='Path to the stimuli folder')
    args = parser.parse_args()
    main(args)
