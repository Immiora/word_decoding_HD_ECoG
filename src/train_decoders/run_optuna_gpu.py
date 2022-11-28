'''

nice python ./train_decoders/run_optuna_gpu.py \
    --task jip_janneke \
    --subject subject1 \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/optuna \
    --model_type densenet \
    --n_trials 5 \
    --n_epochs 1000 \
    --gpu 0 \
    --load_if_exists



'''

import sys
sys.path.insert(0, '.')

import multiprocessing
import optuna
import argparse
import os.path as op
import json
import numpy as np

from os import getpid, makedirs
from optuna import Trial
from contextlib import contextmanager
from utils.arguments import fill_filenames
from train_decoders.optuna_decode_all_hfb_mel import trainer

N_GPUS = 1

class GpuQueue:

    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(N_GPUS)) if N_GPUS > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


class Objective:

    def __init__(self, gpu_queue, args):
        self.gpu_queue = gpu_queue
        self.args = args

    def __call__(self, trial: Trial):
        params = {
            'output_type': trial.suggest_categorical('output_type', ['clean', 'raw']),
            'output_num': trial.suggest_categorical('output_num', [40, 80]), # [20, 40, 80]
            'sr': trial.suggest_categorical('sr', [15, 25, 75]),
            'input_band': trial.suggest_categorical('input_band', ['70-170', '60-300']),
            'input_ref': trial.suggest_categorical('input_ref',['car', 'bipolar']),
            'fragment_len': trial.suggest_float('fragment_len', .16, .36, step=.2),
            'drop_ratio': trial.suggest_loguniform('drop_ratio', 1e-4, 1e-1),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        }

        if self.args.model_type == 'mlp':
            params['mlp_n_blocks'] = trial.suggest_int('mlp_n_blocks', 1, 5)
            params['mlp_n_hidden'] = trial.suggest_int('mlp_n_hidden', 16, 256, step=48)
        elif self.args.model_type == 'densenet':
            params['dense_bottleneck'] = trial.suggest_categorical('dense_bottleneck', [True, False]) # how will this work?
            params['dense_reduce'] = trial.suggest_float('dense_reduce', 0.5, 1, step=.1, log=False)
            params['dense_n_layers'] = trial.suggest_int('dense_n_layers', 10, 16, step=3)
            params['dense_growth_rate'] = trial.suggest_int('dense_growth_rate', 10, 40, step=10)
        elif self.args.model_type == 'seq2seq':
            params['seq_n_enc_layers'] = trial.suggest_int('seq_n_enc_layers', 1, 2)
            params['seq_n_dec_layers'] = trial.suggest_int('seq_n_dec_layers', 1, 2)
            params['seq_bidirectional'] = trial.suggest_categorical('seq_bidirectional', [True, False])

        #with self.gpu_queue.one_gpu_per_process() as gpu_i:

        self.args.trial_id = trial.number
        for key, value in params.items():
            setattr(self.args, key, value)

        args = fill_filenames(self.args)
        args.gpu = args.gpu #'cuda:0'#gpu_i

        best_val_loss = trainer(args)

        return best_val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural decoding with optuna')
    parser.add_argument('--task', '-t', type=str, choices=['jip_janneke', 'hp_reading'])
    parser.add_argument('--subject', '-s', type=str)
    parser.add_argument('--n_trials', '-n', type=int, default=60)
    parser.add_argument('--save_dir', '-p', type=str)
    parser.add_argument('--model_type', '-m', type=str, choices=['mlp', 'densenet', 'seq2seq'])
    parser.add_argument('--n_epochs', '-e', type=int, default=500)
    parser.add_argument('--load_if_exists',  dest='load_if_exists', action='store_true')
    parser.add_argument('--no-load_if_exists', dest='load_if_exists', action='store_false')
    parser.add_argument('--batch_size', '-b', type=int, default=24)
    parser.add_argument('--input_delay', type=float, default=0.0)
    parser.add_argument('--clip_x_value', type=int, default=3)
    parser.add_argument('--clip_t_value', type=int, default=3)
    parser.add_argument('--checkpoint_every', type=int, default=100)
    parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true')
    parser.add_argument('--no-save_checkpoints', dest='save_checkpoints', action='store_false')
    parser.add_argument('--make_plots', dest='make_plots', action='store_true')
    parser.add_argument('--no-make_plots', dest='make_plots', action='store_false')
    parser.add_argument('--save_plots', dest='save_plots', action='store_true')
    parser.add_argument('--no-save_plots', dest='save_plots', action='store_false')
    parser.add_argument('--gpu', type=int, default=0)

    parser.set_defaults(load_if_exists=False)
    parser.set_defaults(save_checkpoints=True)
    parser.set_defaults(make_plots=True)
    parser.set_defaults(save_plots=True)

    args = parser.parse_args()
    args.use_pca = False
    args.n_pcs = 0
    args.early_stop_max = 10000

    seed = np.random.choice(1000, 1) # cannot set constant seed because of parallel distributed setup
    sampler = optuna.samplers.TPESampler(seed=seed)  # Make the sampler behave in a deterministic way. seed=78
    study_name = args.task + '_' + args.subject + '_' + args.model_type
    save_dir = op.join(args.save_dir, args.task, args.subject, args.model_type)
    plot_dir = op.join(args.save_dir, args.task, args.subject, args.model_type).replace('results', 'pics/decoding')
    if not op.exists(save_dir): makedirs(save_dir)
    print(save_dir)
    np.savetxt(op.join(save_dir, 'sampler_seed_process_' + str(getpid()) + '.txt'), seed, fmt='%4d') # save seed to reproduce results

    if args.load_if_exists == False:
        try:
            optuna.study.delete_study(study_name, storage='sqlite:///' + op.join(save_dir, study_name + '.db'))
        except Exception as e:
            pass

    study = optuna.create_study(study_name=study_name,
                                sampler=sampler, storage='sqlite:///' + op.join(save_dir, study_name + '.db'),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10,
                                                                   n_warmup_steps=5,
                                                                   interval_steps=10,
                                                                   n_min_trials=50),
                                direction='minimize',
                                load_if_exists=args.load_if_exists)

    #study.optimize(Objective(GpuQueue(), args), n_trials=args.n_trials, n_jobs=1)
    study.optimize(Objective(None, args), n_trials=args.n_trials, n_jobs=1)
    study.trials_dataframe().to_csv(op.join(save_dir, 'optuna_trials.tsv'), sep='\t')

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(op.join(plot_dir, 'importances'), format='pdf')

    fig = optuna.visualization.plot_contour(study, params=[fig['data'][0].y[-1], fig['data'][0].y[-2]])
    fig.write_image(op.join(plot_dir, 'contour'), format='pdf')

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(op.join(plot_dir, 'opt_history'), format='pdf')

    fig = optuna.visualization.plot_slice(study)
    fig.write_image(op.join(plot_dir, 'param_slice'), format='pdf')

    json.dump(study.best_params,
              open(op.join(save_dir, 'best_params.json'), 'w'),
              indent=4,
              sort_keys=True)

    json.dump(study.best_trial.__dict__,
              open(op.join(save_dir, 'best_trial.json'), 'w'),
              indent=4,
              sort_keys=True,
              default=str)

    print('Best parameters')
    for key, value in study.best_params.items():
        print(key + '=' + str(value) + '\n')