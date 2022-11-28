import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_grid_predictions(epoch, predicted, targets, train_tag, save_pics, plotdir, nrows=5, ncols=5):
    fig = plot_image_grid(predicted, nrows, ncols)
    fig.suptitle(train_tag + ': generated')
    if save_pics:
        plt.savefig(os.path.join(plotdir, train_tag + '_predict_epoch' + str(epoch) + '.png'), dpi=160, transparent=True)
        plt.close()

    fig = plot_image_grid(targets, nrows, ncols)
    fig.suptitle(train_tag + ': real')
    if save_pics:
        plt.savefig(os.path.join(plotdir, train_tag + '_real_epoch' + str(epoch) + '.png'), dpi=160, transparent=True)
        plt.close()


def plot_image_grid(images, nrows, ncols):
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, images):
        ax.imshow(im, origin='lower')

    return fig

def plot_losses(loss_train, loss_val=None, save_pics=False, plotdir=None):
    plt.figure()
    plt.plot(loss_train, color='steelblue', linestyle='-', marker='.', label='train')
    plt.title('Train loss')
    if loss_val is not None:
        plt.plot(loss_val, color='darkred', linestyle='-', marker='.', label='validation')
    plt.title('Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.tight_layout()
    if save_pics:
        plt.savefig(os.path.join(plotdir, 'loss.pdf'), dpi=160, transparent=True)
        plt.close()

def plot_scatter_predictions_targets(targets, predictions, train_tag, save_pics=False, plotdir=None):
    """
    inputs: 2d arrays: time x features for targets and predictions
    """

    fig = plt.figure(figsize=(10., 10.))
    if targets.shape[-1] == 20:
        nrows = 4
        ncols = int(20 / nrows)
    elif targets.shape[-1] == 40:
        nrows = 5
        ncols = int(40 / nrows)
    elif targets.shape[-1] == 80:
        nrows = 8
        ncols = int(80 / nrows)
    else:
        raise ValueError

    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, t, p in zip(grid, targets.T, predictions.T):
        ax.scatter(t, p)

    if save_pics:
        plt.savefig(os.path.join(plotdir, train_tag + '_scatter_pred_targets.pdf'), dpi=160, transparent=True)
        plt.close()


def plot_eval_metric_box(metric, df, plotdir=None, by=None):
    bx = df.boxplot(by=by) if by is not None else df.boxplot()
    bx.set_ylabel(metric)
    bx.set_ylim(-0.1, 1.1)
    fig = bx.get_figure()
    if plotdir is not None:
        plt.savefig(os.path.join(plotdir, 'eval_' + metric + '.pdf'), dpi=160, transparent=True)
        plt.close()


def plot_eval_metric_mean(metric, df, plotdir=None, by=None):
    bx = df.plot(kind='bar')
    bx.set_ylabel(metric)
    bx.set_ylim(-0.1, 1.1)
    fig = bx.get_figure()
    if plotdir is not None:
        plt.savefig(os.path.join(plotdir, 'eval_' + metric + '.pdf'), dpi=160, transparent=True)
        plt.close()

def plot_electrode_grid(grid, subject, plotdir=None):
    plt.imshow(grid)
    if plotdir is not None:
        plt.savefig(os.path.join(plotdir, 'electrode_grid_' + subject + '.pdf'), dpi=160, transparent=True)
        plt.close()

def get_model_cmap(model):
    if model == 'mlp':
        color = 'Blues'
    elif model == 'densenet':
        color = 'Oranges'
    elif model == 'seq2seq':
        color = 'Greens'
    else:
        raise ValueError
    return color