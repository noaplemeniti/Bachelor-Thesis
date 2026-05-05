import yaml
import os
import numpy as np
import torch
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def tensor_img_to_npimg(tensor_img):
    if not (torch.is_tensor(tensor_img) and tensor_img.ndimension() == 3):
        raise NotImplementedError("Only tensors with dimension CxHxW are supported.")
    npimg = np.transpose(tensor_img.numpy(), (1, 2, 0))
    npimg = npimg.squeeze()
    assert isinstance(npimg, np.ndarray) and (npimg.ndim in {2, 3})
    return npimg


def normalize(x):
    return x.mul_(2).add_(-1)


def deprocess(img):
    return img.add_(1).div_(2)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    _, _, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    return torch.nn.ZeroPad2d(paddings)(images)


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)

    unfold = torch.nn.Unfold(
        kernel_size=ksizes,
        dilation=rates,
        padding=0,
        stride=strides,
    )
    return unfold(images)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)
    
def get_model_list(dirname, key, iteration=0):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    if iteration == 0:
        last_model_name = gen_models[-1]
    else:
        for model_name in gen_models:
            if '{:0>8d}'.format(iteration) in model_name:
                return model_name
        raise ValueError('Not found models with this iteration')
    return last_model_name

def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = config['spatial_discounting_gamma']
    height, width = config['mask_shape']
    shape = [1, 1, height, width]
    if config['discounted_mask']:
        mask_values = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                mask_values[i, j] = max(
                    gamma ** min(i, height - i),
                    gamma ** min(j, width - j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 0)
    else:
        mask_values = np.ones(shape)
    spatial_discounting_mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
    if config['cuda']:
        spatial_discounting_mask_tensor = spatial_discounting_mask_tensor.cuda()
    return spatial_discounting_mask_tensor


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def flow_to_image(flow):
    out = []
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknown = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknown] = 0
        v[idxunknown] = 0
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        out.append(compute_color(u, v))
    return np.float32(np.uint8(out))


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nan_idx = np.isnan(u) | np.isnan(v)
    u[nan_idx] = 0
    v[nan_idx] = 0
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[np.logical_not(idx)] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nan_idx)))
    return img


def make_color_wheel():
    ry, yg, gc, cb, bm, mr = (15, 6, 4, 11, 13, 6)
    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros([ncols, 3])
    col = 0
    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.transpose(np.floor(255 * np.arange(0, ry) / ry))
    col += ry
    colorwheel[col:col + yg, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, yg) / yg))
    colorwheel[col:col + yg, 1] = 255
    col += yg
    colorwheel[col:col + gc, 1] = 255
    colorwheel[col:col + gc, 2] = np.transpose(np.floor(255 * np.arange(0, gc) / gc))
    col += gc
    colorwheel[col:col + cb, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, cb) / cb))
    colorwheel[col:col + cb, 2] = 255
    col += cb
    colorwheel[col:col + bm, 2] = 255
    colorwheel[col:col + bm, 0] = np.transpose(np.floor(255 * np.arange(0, bm) / bm))
    col += bm
    colorwheel[col:col + mr, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, mr) / mr))
    colorwheel[col:col + mr, 0] = 255
    return colorwheel


def is_image_file(filename):
    img_extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in img_extensions)


def pil_to_float_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.tensor(array.tolist(), dtype=torch.float32).permute(2, 0, 1)


def tensor_to_pil_image(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0)
    array = (tensor.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)
