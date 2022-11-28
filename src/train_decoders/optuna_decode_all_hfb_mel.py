
"""
python 3.8
'torch' conda environment


python ./train_decoders/optuna_decode_all_hfb_mel.py \
    -i /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_hfb_jip_janneke_car_70-170_25Hz.npy \
    -o /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_audio_jip_janneke_clean_nfft882_hop882_mel40_25Hz.npy \
    --input_sr 25. \
    --output_sr 25. \
    --input_ref car \
    --input_band 70-170 \
    --input_mean /Fridge/users/julia/project_decoding_jip_janneke/data//subject1_hfb_jip_janneke_car_70-170_25Hz_rest_precomputed_mean.npy \
    --input_std /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_hfb_jip_janneke_car_70-170_25Hz_rest_precomputed_std.npy \
    --output_mean /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_audio_jip_janneke_clean_nfft882_hop882_mel40_25Hz_mean.npy \
    --output_std /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_audio_jip_janneke_clean_nfft882_hop882_mel40_25Hz_std.npy \
    --save_dir /Fridge/users/julia/project_decoding_jip_janneke/results/old/subject1/plot_model \
    --fragments /Fridge/users/julia/project_decoding_jip_janneke/data/subject1/subject1_contexts_jip_janneke_25Hz_step0.04s_window0.36s_speech_only.csv \
    --fragment_len .36 \
    --model_type mlp \
    --mlp_n_blocks 3 \
    --mlp_n_hidden 128 \
    --drop_ratio .0 \
    --n_epochs 1 \
    --learning_rate 8e-4

    --no-dense_bottleneck
    --dense_reduce .7
    --dense_n_layers 10
    --dense_growth_rate 40

    --seq_n_enc_layers 1
    --seq_n_dec_layers 1
    --seq_bidirectional
"""


import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import tqdm
import argparse
import os.path

from utils.training import preprocess_x, preprocess_t, denormalize
from utils.plots import plot_grid_predictions, plot_losses
from utils.general import make_save_dir
from utils.datasets import BrainDataset, split_data, load_data, get_moments
from models.select_model import select_model
from torch.utils.data import DataLoader

def trainer(args):

    # get data
    dataset = BrainDataset(args.input_file, args.output_file, args.fragments,
                           args.input_sr, args.output_sr, args.fragment_len, args.input_delay)
    trainset, valset, testset = split_data(dataset)

    # compute mean and std on train data
    x_mean, x_std, t_mean, t_std, pca = get_moments(trainset, args.input_mean, args.input_std,
                                                    args.output_mean, args.output_std, args.use_pca,
                                                    args.n_pcs, can_write=True)

    # data loaders
    train_loader, val_loader, test_loader = load_data(trainset, valset, testset, args.batch_size)

    # save info
    n_out = t_mean.shape[-1]
    args.save_dir = os.path.join(args.save_dir, args.model_type + \
                           '_sr' + str(args.input_sr) + \
                           '_' + str(args.input_ref) + \
                           '_hfb' + str(args.input_band) + \
                           '_flen' + str(args.fragment_len) +\
                           '_lr' + str(round(args.learning_rate, 4)) + \
                           '_nout' + str(n_out) + \
                           '_drop' + str(round(args.drop_ratio, 4)) +
                           '_e' + str(args.n_epochs) + '.')
    args.save_dir = make_save_dir(args.save_dir)

    with open(os.path.join(args.save_dir, 'input_parameters.txt'), 'w') as f:
        for key, value in args._get_kwargs():
            f.write(key + '=' + str(value) + '\n')

    plotdir = None
    if args.save_plots:
        plotdir = args.save_dir.replace('results', 'pics/decoding')
        if not os.path.isdir(plotdir): os.makedirs(plotdir)

    # set up torch
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu") # "cuda:0"
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(device)
    print('Device: ', device)
    torch.manual_seed(0)

    # set up model
    net = select_model(args, train_loader, device)
    with open(os.path.join(args.save_dir, 'model_summary.txt'), 'w') as f:
        print(net, file=f)
    # torch.onnx.export(net,
    #                   preprocess_x(next(iter(train_loader))[0], x_mean, x_std, args.clip_x_value, device, pca=None),
    #                   os.path.join(args.save_dir, 'model.onnx'),
    #                   input_names=['brain'],
    #                   output_names=['spectra'])
    #                   #opset_version = 11) # for densenet, otherwise Failed to export an ONNX attribute 'onnx::Sub', since it's not constant, please try to make things (e.g., kernel size) static if possible
    #                                       # https://github.com/onnx/tutorials/issues/137

    # optimizer and loss
    adam = torch.optim.Adam(net.parameters(), args.learning_rate)
    mse = torch.nn.MSELoss().to(device)

    # train
    L = np.zeros(args.n_epochs)
    L_val = np.zeros(args.n_epochs)
    min_val_loss = 10000
    early_stop_counter = -1

    for e in tqdm.trange(args.n_epochs):
        net = net.train()
        for (x, t, ith) in train_loader:
            out = net(preprocess_x(x, x_mean, x_std, args.clip_x_value, device, pca=None))
            loss = mse(out, preprocess_t(t, t_mean, t_std, args.clip_t_value, device))
            L[e] += loss.cpu().detach().numpy()
            #print(loss)
            net.zero_grad()
            loss.backward()
            adam.step()
        L[e] /= len(train_loader)

        net = net.eval()
        predicted_val_all = []
        targets_val_all = []
        for b, (x_val, t_val, ith_val) in enumerate(val_loader):
            out_val = net(preprocess_x(x_val, x_mean, x_std, args.clip_x_value, device, pca=None))
            loss_val = torch.nn.MSELoss().cuda()(out_val, preprocess_t(t_val, t_mean, t_std, args.clip_t_value, device))
            L_val[e] += loss_val.cpu().detach().numpy()
            #print('Validation loss: ', loss_val)

            predicted_val_all.append(
                denormalize(out_val.detach(), t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy())
            targets_val_all.append(t_val.transpose(1, 2).cpu().detach().numpy())

        L_val[e] /= len(val_loader)

        predicted_all, targets_all = [], []
        for (x, t, ith) in DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0):
            out = net(preprocess_x(x, x_mean, x_std, args.clip_x_value, device, pca=None))
            predicted_all.append(denormalize(out.detach(), t_mean, t_std, args.clip_t_value, device).cpu().detach().numpy())
            targets_all.append(t.transpose(1, 2).cpu().detach().numpy())

        if L_val[e] < min_val_loss:
            #print('Minimal val loss reached')
            torch.save({
                'epoch': e,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': adam.state_dict(),
                'loss': L[e],
            }, os.path.join(args.save_dir, 'min_val_loss.pth'))

            if args.make_plots:
                plot_grid_predictions(e,
                  np.concatenate(predicted_val_all).squeeze()[:int(4*args.input_sr)].reshape(1, -1, n_out).swapaxes(1, 2),
                  np.concatenate(targets_val_all).squeeze()[:int(4*args.input_sr)].reshape(1, -1, n_out).swapaxes(1, 2),
                  train_tag='min_val_concat', save_pics=args.save_plots, plotdir=plotdir, nrows=1, ncols=1)
            min_val_loss = L_val[e]
            early_stop_counter = -1

        if args.save_checkpoints:
            if (e + 1) % args.checkpoint_every == 0:
                np.save(os.path.join(args.save_dir, 'loss_train.npy'), L[L > 0])
                np.save(os.path.join(args.save_dir, 'loss_val.npy'), L_val[L_val > 0])
                #print('Saving interim examples epoch: ' + str(e))
                if args.make_plots:
                    plot_grid_predictions(e,
                      np.concatenate(predicted_all).squeeze()[:int(20*args.input_sr)].reshape(5, -1, n_out).swapaxes(1, 2),
                      np.concatenate(targets_all).squeeze()[:int(20*args.input_sr)].reshape(5, -1, n_out).swapaxes(1, 2),
                      train_tag='train_concat', save_pics=args.save_plots, plotdir=plotdir, nrows=5, ncols=1)

                    plot_grid_predictions(e,
                      np.concatenate(predicted_val_all).squeeze()[:int(4*args.input_sr)].reshape(1, -1, n_out).swapaxes(1, 2),
                      np.concatenate(targets_val_all).squeeze()[:int(4*args.input_sr)].reshape(1, -1, n_out).swapaxes(1, 2),
                      train_tag='val_concat', save_pics=args.save_plots, plotdir=plotdir, nrows=1, ncols=1)
                torch.save({
                    'epoch': e,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': adam.state_dict(),
                    'loss': L[e],
                }, os.path.join(args.save_dir, 'checkpoint_' + str(e) + '.pth'))

        early_stop_counter += 1
        if early_stop_counter >= args.early_stop_max:
            args.n_epochs = e
            L = L[:args.n_epochs]
            L_val = L_val[:args.n_epochs]
            break

    if args.make_plots:
        plot_grid_predictions(args.n_epochs,
                  np.concatenate(predicted_all).squeeze()[:int(20*args.input_sr)].reshape(5, -1, n_out).swapaxes(1, 2),
                  np.concatenate(targets_all).squeeze()[:int(20*args.input_sr)].reshape(5, -1, n_out).swapaxes(1, 2),
                  train_tag='train_concat', save_pics=args.save_plots, plotdir=plotdir, nrows=5, ncols=1)
        plot_grid_predictions(args.n_epochs,
                  np.concatenate(predicted_val_all).squeeze()[:int(4*args.input_sr)].reshape(1, -1, n_out).swapaxes(1, 2),
                  np.concatenate(targets_val_all).squeeze()[:int(4*args.input_sr)].reshape(1, -1, n_out).swapaxes(1, 2),
                  train_tag='val_concat', save_pics=args.save_plots, plotdir=plotdir, nrows=1, ncols=1)
        plot_losses(L, L_val, save_pics=args.save_plots, plotdir=plotdir)

    torch.save({
        'epoch': args.n_epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': adam.state_dict(),
        'loss': L[-1],
    }, os.path.join(args.save_dir, 'checkpoint_' + str(args.n_epochs) + '.pth'))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return L_val[-1]



##
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural decoding Vanilla')

    # model
    parser.add_argument('--model_type', type=str, choices=['densenet', 'seq2seq', 'mlp', 'resnet'])
    parser.add_argument('--drop_ratio', type=float, default=0.0)

    parser.add_argument('--mlp_n_blocks', type=int, default=3)
    parser.add_argument('--mlp_n_hidden', type=int, default=64)

    parser.add_argument('--dense_bottleneck', dest='dense_bottleneck', action='store_true')
    parser.add_argument('--no-dense_bottleneck', dest='dense_bottleneck', action='store_false')
    parser.add_argument('--dense_reduce', type=float, default=1.)
    parser.add_argument('--dense_n_layers', type=int, default=10)
    parser.add_argument('--dense_growth_rate', type=int, default=10)

    parser.add_argument('--seq_n_enc_layers', type=int, default=1)
    parser.add_argument('--seq_n_dec_layers', type=int, default=1)
    parser.add_argument('--seq_bidirectional', dest='seq_bidirectional', action='store_true')
    parser.add_argument('--no-seq_bidirectional', dest='seq_bidirectional', action='store_false')

    # data
    parser.add_argument('--input_file', '-i', type=str)
    parser.add_argument('--input_sr', '-isr', type=float)
    parser.add_argument('--input_ref', type=str)
    parser.add_argument('--input_band', type=str)
    parser.add_argument('--input_mean', '-im', type=str)
    parser.add_argument('--input_std', '-is', type=str)
    parser.add_argument('--input_delay', type=float, default=0.0)
    parser.add_argument('--n_pcs', type=int, default=100)
    parser.add_argument('--use_pca', dest='use_pca', action='store_true')
    parser.add_argument('--no-use_pca', dest='use_pca', action='store_false')

    parser.add_argument('--output_file', '-o', type=str)
    parser.add_argument('--output_sr', '-osr', type=float)
    parser.add_argument('--output_mean', '-om', type=str)
    parser.add_argument('--output_std', '-os', type=str)

    parser.add_argument('--fragments', type=str)
    parser.add_argument('--fragment_len', type=float)

    # training
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--checkpoint_every', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--early_stop_max', type=int, default=10000)
    parser.add_argument('--clip_x_value', type=int, default=3)
    parser.add_argument('--clip_t_value', type=int, default=3)
    parser.add_argument('--gpu', type=int, default=0)

    # other
    parser.add_argument('--save_dir', '-s', type=str)
    parser.add_argument('--save_checkpoints', dest='save_checkpoints', action='store_true')
    parser.add_argument('--no-save_checkpoints', dest='save_checkpoints', action='store_false')
    parser.add_argument('--make_plots', dest='make_plots', action='store_true')
    parser.add_argument('--no-make_plots', dest='make_plots', action='store_false')
    parser.add_argument('--save_plots', dest='save_plots', action='store_true')
    parser.add_argument('--no-save_plots', dest='save_plots', action='store_false')

    # binary defaults
    parser.set_defaults(dense_bottleneck=False)
    parser.set_defaults(seq_bidirectional=False)
    parser.set_defaults(use_pca=False)
    parser.set_defaults(save_checkpoints=True)
    parser.set_defaults(make_plots=True)
    parser.set_defaults(save_plots=True)
    args = parser.parse_args()

    trainer(args)


## gradient viz
# after loss.backward()
#
# ave_grads = []
# layers = []
# for n, p in net.named_parameters():
#     if (p.requires_grad) and ("bias" not in n):
#         layers.append(n)
#         ave_grads.append(p.grad.abs().mean())
# plt.plot(ave_grads, alpha=0.3, color="b")
# plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
# plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
# plt.xlim(xmin=0, xmax=len(ave_grads))
# plt.xlabel("Layers")
# plt.ylabel("average gradient")
# plt.title("Gradient flow")
# plt.grid(True)


