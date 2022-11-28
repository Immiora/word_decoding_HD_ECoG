'''
plot spectrogram of audio and brain data for the contamination analysis

python figures/figureS2a.py \
    --task jip_janneke \
    --subject_code zk0v \
    --ref noref car bipolar \
    --plot_dir /Fridge/users/julia/project_decoding_jip_janneke/pics/figures

'''


import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import os.path as op
import argparse
import json
import pandas as pd
from utils.private.datasets import get_subjects_by_code
from utils.general import sec2ind, load_csv_data

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

##
def main(args):

    data_dir = op.join('/Fridge/users/julia/project_decoding_' + args.task, 'data')
    tval_speech_dir = '/Fridge/users/julia/project_decoding_jip_janneke/results/ttest_speech_nonspeech'
    sr_input = 2000
    sr_output = 16000
    st_s = 60
    en_s = 70

    for isubject, subject in enumerate(args.subject):
        print(subject)

        for ref in args.ref:
            input_file = op.join(data_dir, subject, 'time_' + args.task + '_ecog_' + ref + '_nofreq_noz_2000Hz.csv')
            input_data, input_labels = load_csv_data(input_file)

            output_info = json.loads(open(op.join(data_dir, subject, subject + '_audio_' + args.task + '.json'), 'r').read())
            output_file = output_info['raw']
            output_data, sr_raw = librosa.load(output_file, sr=None)
            output_data_res = librosa.resample(output_data, sr_raw, sr_output)

            onset_input = sec2ind(st_s, sr_input)
            offset_input = sec2ind(en_s, sr_input)
            onset_output = sec2ind(st_s, sr_output)
            offset_output = sec2ind(en_s, sr_output)

            # load t-values for plotting 1e correlation profile
            # determine electrode based on highest t-value speech vs music (load csv with results) (for noref and car)
            speech_tvals = pd.read_csv(op.join(tval_speech_dir, subject + '.csv'), header=0)
            el = speech_tvals.sort_values(by=['t-value'], ascending=False).reset_index(drop=True).loc[0, 'electrodes']
            if ref == 'bipolar':
                el = [a for a,b in enumerate(input_labels.values) if 'chan'+str(el) in b ][0]
            E = librosa.amplitude_to_db(np.abs(librosa.stft(input_data[onset_input:offset_input, el], n_fft=512)), ref=np.max)
            A = librosa.amplitude_to_db(np.abs(librosa.stft(output_data_res[onset_output:offset_output], n_fft=4096)), ref=np.max)

            plt.figure(figsize=(10, 8), dpi=160)
            plt.subplot(211)
            librosa.display.specshow(A, y_axis='linear', sr=sr_output, hop_length=4096/4, x_axis='time', cmap='pink_r')
            plt.ylim(0, 600)
            plt.clim(-70, 0)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-frequency power spectrogram')

            # use vmax=-70 in specshow for zk0v and comment out clt.clim
            plt.subplot(212)

            if args.subject_code[isubject] == 'zk0v' and ref in ['car', 'bipolar']:
                librosa.display.specshow(E, y_axis='linear', sr=sr_input, vmax=-70,
                                         hop_length=512/4, x_axis='time', cmap='pink_r')
            else:
                librosa.display.specshow(E, y_axis='linear', sr=sr_input,
                                         hop_length=512/4, x_axis='time', cmap='pink_r')
                plt.clim(-70, 0)
            plt.ylim(0, 600)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Log-frequency power spectrogram, E' + str(el))
            plt.tight_layout()
            plt.savefig(op.join(args.plot_dir,
                    'figS2a_audio_contamination_spectra_el' +
                                str(el) + '_' + ref + '_' + subject + '_' + str(st_s) + '_' + str(en_s) + '.pdf'),
                                 dpi=160, transparent=True)
            plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--task', '-t', type=str)
    parser.add_argument('--subject_code', '-s', type=str,  nargs="+",
                        choices=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        default=['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop'],
                        help='Subject to run')
    parser.add_argument('--ref', '-r', type=str,  nargs="+", choices=['bipolar', 'car', 'noref'])
    parser.add_argument('--plot_dir', '-p', type=str, help='Plot directory', default='')
    args = parser.parse_args()
    args.subject = get_subjects_by_code(args.subject_code)
    main(args)
