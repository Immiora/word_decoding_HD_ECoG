# python 3.8
# 'parallel_wavegan' conda environment

"""
Generate sounds for the specified checkpoint of the decoder
Uses speech_only fragments, not fragments for classification (all 1s long)

Update 04/06/2021:
    - updated version to accomodate changes in the decoder scripts
    - can specify varying input and output files (along with moments), different fragments files
        (with train/test/val attributions) and different paths for the GAN model if needed for the decoder.

    - can only produce .93 fragments from onsets in the fragments file. Need to set up a separate routine to generate
        train/validation/test small speech only, non-overllaping fragments or full sentences

Update 02/06/2021:
    - instead of only using min_val_loss, pass full path to the checkpoint

Update 27/07/2021:
    - changed model_type, now take densenet, seq2seq, resnet or mpl;
    - gan_z temporarily does not work

Update 09/08/2021:
    - changes to accomodate the optimal decoder project
    - input folder with .pth model checkpoints and input_parameters.txt, take parameters from there
    - input checkpoint filename


Example run:
python synthesize_audio/synth_speech_targets_reconstructed.py \
    -p /Fridge/users/julia/project_decoding_jip_janneke/results/optuna/jip_janneke/subject1/densenet_launch1/14/densenet_sr15_car_hfb60-300_flen0.36_lr0.0098_nout20_drop0.0004_e1000. \
    -c checkpoint_999 \
    --target_set test \
    --synth_targets


"""
import sys
sys.path.insert(0, '.')

import os
import argparse
from subprocess import check_call
import numpy as np
import librosa

from torch.utils.data import Dataset, DataLoader
from utils.general import write_hdf5
from utils.training import *
from utils.arguments import load_params
from utils.datasets import BrainDataset, split_data, get_moments
from models.select_model import select_model


def synthesize(args):


    # set up output
    out_dir = args.save_dir
    if out_dir == '':
        out_dir = args.model_path
    synth_dir = os.path.join(out_dir, 'parallel_wavegan')
    dump_dir = os.path.join(synth_dir, 'dump', 'norm')
    root_dir = os.path.join(synth_dir, 'dump', 'raw')

    # get data
    dataset = BrainDataset(args.input_file, args.output_file, args.fragments,
                           args.input_sr, args.output_sr, args.fragment_len, args.input_delay)
    trainset, valset, testset = split_data(dataset)

    # compute mean and std on train data
    x_mean, x_std, t_mean, t_std, pca = get_moments(trainset, args.input_mean, args.input_std,
                                                    args.output_mean, args.output_std, args.use_pca,
                                                    args.n_pcs, can_write=False)

    # set up torch
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    if torch.cuda.is_available() and args.use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(device)
    print('Device: ', device)

    # data loaders
    target_dataset = torch.utils.data.Subset(dataset, dataset.fragments.index[dataset.fragments['subset']==args.target_set].tolist())
    target_loader = DataLoader(target_dataset, batch_size=len(target_dataset), shuffle=False, num_workers=0)
    if len(target_loader) > 128: target_loader = target_loader[:128]

    # set up models
    model = select_model(args, target_loader, device)

    model_path = os.path.join(args.model_path, args.checkpoint + '.pth')
    assert os.path.exists(model_path), 'Model path does not exist: ' + model_path
    saved = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(saved['model_state_dict'])

    # pass through the model
    print('Computing predictions')
    predictions, targets = [], []
    for (x, t, ith) in target_loader:
        out = model(preprocess_x(x, x_mean, x_std, args.clip_x_value, device, pca=None))
        predictions.append(denormalize(out.detach(), t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy())
        targets.append(t.transpose(1, 2).cpu().detach().numpy())
    predictions = np.concatenate(predictions).squeeze()
    targets = np.concatenate(targets).squeeze()

    # upsample
    # round(predictions.shape[0]/args.sr*86.13)
    rt = 'scipy' # less artifact on last mel and timepoints than default
    predictions = librosa.resample(librosa.resample(predictions.T, args.output_sr, 86.13, res_type=rt).T, args.output_num, 80, res_type=rt)
    targets = librosa.resample(librosa.resample(targets.T, args.output_sr, 86.13, res_type=rt).T, args.output_num, 80, res_type=rt)

    # generate sound
    for d in [synth_dir, dump_dir, root_dir]:
        if not os.path.isdir(d): os.makedirs(d)
        wavegan_dump_fname = os.path.join(root_dir, args.model_type + '_' + args.target_set + '.h5')
        write_hdf5(wavegan_dump_fname, "feats", predictions.astype(np.float32))
        if args.synth_targets:
            wavegan_dump_fname = os.path.join(root_dir, 'targets_' + args.target_set + '.h5')
            write_hdf5(wavegan_dump_fname, "feats", targets.astype(np.float32))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # run synthesis
    # try source activate environment-name &&
    rc = check_call(['parallel-wavegan-normalize',
                     '--config', '/Fridge/users/julia/project_speechGANs/parallel_wavegan/pretrained_models/wavegan_v1/config.yml',
                     '--rootdir', root_dir,
                     '--dumpdir', dump_dir,
                     '--stats', '/Fridge/users/julia/project_speechGANs/parallel_wavegan/pretrained_models/wavegan_v1/stats.h5',
                     '--skip-wav-copy'])
    print(rc)

    rc = check_call(['parallel-wavegan-decode',
                     '--checkpoint', '/Fridge/users/julia/project_speechGANs/parallel_wavegan/pretrained_models/wavegan_v1/checkpoint-400000steps.pkl',
                     '--dumpdir', dump_dir,
                     '--outdir', synth_dir])
    print(rc)



## load normalization parameters
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parameters for sound generation')
    parser.add_argument('--model_path', '-p', type=str, help='Directory with model')
    parser.add_argument('--checkpoint', '-c', type=str, help='Checkpoint', default='min_val_loss')
    parser.add_argument('--target_set', '-t', type=str, help='Target set', default='test')
    parser.add_argument('--save_dir', '-s', type=str, help='Output directory', default='')
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true')
    parser.add_argument('--no-use_cuda', dest='use_cuda', action='store_false')
    parser.add_argument('--synth_targets', dest='synth_targets', action='store_true')
    parser.add_argument('--no-synth_targets', dest='synth_targets', action='store_false')

    parser.set_defaults(use_cuda=True)
    parser.set_defaults(synth_targets=True)
    args = parser.parse_args()
    save_dir = args.save_dir
    use_cuda = args.use_cuda

    print(args.model_path)

    args = load_params(args)
    args.save_dir = save_dir # overwrite from loaded params
    args.use_cuda = use_cuda # overwrite from loaded params

    synthesize(args)
