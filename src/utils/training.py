
import torch


def normalize(x, mean, std):
    if type(x) == torch.Tensor:
        return x.sub(mean[None, :, None]).div(std[None, :, None]).type(torch.FloatTensor)
    else:
        raise ValueError('Input must be torch.Tensor')


def preprocess_x(x, x_mean, x_std, clip_x_value, DEVICE, pca):
    x_norm = normalize(x.transpose(1, 2), x_mean, x_std)
    if pca is not None:
        #x_norm = pca.transform(x_norm)
        x_norm = x_norm.transpose(1, 2)
        #shape = x_norm.shape
        #x_norm = x_norm.view(-1, pca.components_.shape[0])
        if pca.mean_ is not None:
            x_norm = x_norm - pca.mean_[None, None, :]
        x_norm = torch.matmul(x_norm, torch.Tensor(pca.components_.T))
        #x_norm = x_norm.reshape(shape)
        x_norm = x_norm.transpose(1, 2)
    if clip_x_value is not None:
        x_norm /= clip_x_value
        x_norm = torch.clamp(x_norm, -1., 1.)
    return torch.autograd.Variable(x_norm, requires_grad=False).to(DEVICE)


def preprocess_t(t, t_mean, t_std, clip_t_value, DEVICE):
    t_norm = normalize(t.transpose(1, 2), t_mean, t_std)
    if clip_t_value is not None:
        t_norm /= clip_t_value
        t_norm = torch.clamp(t_norm, -1., 1.)
    return torch.autograd.Variable(t_norm, requires_grad=False).to(DEVICE)


def denormalize(t_norm, t_mean, t_std, clip_t_value, DEVICE):
    if clip_t_value is not None:
        t_norm *= clip_t_value
    t = t_norm.mul(t_std[None, :, None].to(DEVICE)).add(t_mean[None, :, None].to(DEVICE))
    return t
