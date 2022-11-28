import numpy as np
import os.path as op
import copy
import json
from collections import OrderedDict
from utils.private.datasets import get_subjects_by_code

def fill_filenames(args):

    keys = ['task', 'subject', 'sr', 'output_type', 'output_num', 'input_band', 'input_ref', 'input_win', 'filename']
    info = OrderedDict.fromkeys(keys)

    info['task'] = ['jip_janneke', 'hp_reading']
    info['subject_codes'] = ['sgf0', 'l4cf', 'jvu9', 'zk0v', 'xoop']
    info['subject'] = get_subjects_by_code(info['subject_codes'])
    info['sr'] = [15, 25, 75]
    info['output_type'] = ['raw', 'clean']
    info['output_num'] = [20, 40, 80]
    info['input_band'] = ['70-170', '60-300']
    info['input_ref'] = ['car', 'bipolar']
    info['input_win'] = [.16, .36] # numbers will differ on sr, find closest number to these

    nffts = {}
    nffts[15] = 1470
    nffts[25] = 882
    nffts[75] = 525

    hops = nffts.copy()
    hops[75] = 294

    steps = {}
    steps[15] = .0667
    steps[25] = .04
    steps[75] = .0133

    windows = {}
    windows[15] = np.array([.1334, .3335])
    windows[25] = np.array([.16, .36])
    windows[75] = np.array([.1596, .3591])


    out = copy.deepcopy(args)

    dir_name = op.join('/Fridge/users/julia/project_decoding_' + args.task, 'data', args.subject)
    out.input_file = op.join(dir_name,
                         args.subject + '_hfb_' +\
                         args.task + '_' +\
                         args.input_ref + '_' +\
                         args.input_band + '_' + \
                         str(args.sr) + 'Hz.npy')
    out.input_mean = out.input_file.replace('.npy', '_rest_precomputed_mean.npy')
    out.input_std = out.input_file.replace('.npy', '_rest_precomputed_std.npy')

    out.output_file = op.join(dir_name,
                         args.subject + '_audio_' +\
                         args.task + '_' +\
                         args.output_type + '_' +\
                         'nfft' + str(nffts[args.sr]) + '_' + \
                         'hop' + str(hops[args.sr]) + '_' + \
                         'mel' + str(args.output_num) + '_' + \
                          str(args.sr) + 'Hz.npy')
    out.output_mean = out.output_file.replace('.npy', '_mean.npy')
    out.output_std = out.output_file.replace('.npy', '_std.npy')

    for ifile in [out.input_file, out.input_mean, out.input_std, out.output_file]:
        assert op.exists(ifile), 'File does not exist: ' + ifile

    out.fragments = op.join(dir_name,
                            args.subject + '_contexts_' + \
                            args.task + '_' + \
                            str(args.sr) + 'Hz_' +\
                            'step' + str(steps[args.sr]) + 's_' +\
                            'window' + str(windows[args.sr][np.argmin(np.abs(windows[args.sr]-args.fragment_len))]) +\
                            's_speech_only.csv')

    out.subsets_path = json.load(open(out.fragments.replace('_speech_only.csv', '.json'), 'r'))['subsets_path']

    assert op.exists(out.fragments), 'File does not exist: ' + out.fragments

    out.input_sr = args.sr
    out.output_sr = args.sr

    out.save_dir = op.join(args.save_dir, args.task, args.subject, args.model_type, str(args.trial_id))

    return out


def load_params(args):
    if op.exists(op.join(args.model_path, 'input_parameters.txt')):
        params = {}
        with open(op.join(args.model_path, 'input_parameters.txt'), 'r') as f:
            content = f.read().splitlines()
            for line in content:
                temp = line.split('=')
                params[temp[0]] = temp[1]

        # fix variable type: int, float, bool
        for key, value in params.items():
            if key in ['batch_size', 'checkpoint_every', 'clip_t_value', 'clip_x_value',
                       'dense_growth_rate', 'dense_n_layers', 'early_stop_max', 'mlp_n_blocks',
                       'mlp_n_hidden', 'seq_n_dec_layers', 'seq_n_enc_layers',
                       'n_epochs', 'n_pcs', 'output_num', 'sr', 'input_sr', 'output_sr', 'gpu']:
                params[key] = int(value)
            elif key in ['dense_reduce', 'drop_ratio', 'fragment_len', 'input_delay', 'learning_rate']:
                params[key] = float(value)
            elif key in ['dense_bottleneck', 'make_plots', 'save_checkpoints', 'save_plots',
                         'use_pca', 'seq_bidirectional']:
                if value.lower() in ['true', '1']:
                    params[key] = True
                elif value.lower() in ['false', '0']:
                    params[key] = False
                else:
                    raise ValueError

        for key, value in params.items():
            setattr(args, key, value)

        return args

    else:
        print(op.join(args.model_path, 'input_parameters.txt') + ' does not exist')
        raise ValueError()
