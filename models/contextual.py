from typing import List
import torch

import torch.nn.functional as F

from collections import namedtuple

import torch.nn as nn
import torchvision.models.vgg as vgg


LOSS_TYPES = ['cosine', 'l1', 'l2']


class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2',
                           'relu3_4', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2,
                          h_relu3_4, h_relu4_4, h_relu5_4)

        return out


def contextual_loss(x: torch.Tensor,
                    y: torch.Tensor,
                    band_width: float = 0.5,
                    loss_type: str = 'cosine'):
    """
    Computes contextual loss between x and y.
    The most of this code is copied from
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da.

    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.

    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    N, C, H, W = x.size()

    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(x, y)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(x, y)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(x, y)

    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)
    cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)

    return cx_loss


# TODO: Operation check
def contextual_bilateral_loss(x: torch.Tensor,
                              y: torch.Tensor,
                              weight_sp: float = 0.1,
                              band_width: float = 1.,
                              loss_type: str = 'cosine'):
    """
    Computes Contextual Bilateral (CoBi) Loss between x and y,
        proposed in https://arxiv.org/pdf/1905.05169.pdf.

    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W).
    y : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.

    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper).
    k_arg_max_NC : torch.Tensor
        indices to maximize similarity over channels.
    """

    assert x.size() == y.size(), 'input tensor must have the same size.'
    assert loss_type in LOSS_TYPES, f'select a loss type from {LOSS_TYPES}.'

    # spatial loss
    grid = compute_meshgrid(x.shape).to(x.device)
    dist_raw = compute_l2_distance(grid, grid)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_sp = compute_cx(dist_tilde, band_width)

    # feature loss
    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(x, y)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(x, y)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(x, y)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_feat = compute_cx(dist_tilde, band_width)

    # combined loss
    cx_combine = (1. - weight_sp) * cx_feat + weight_sp * cx_sp

    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)

    cx = k_max_NC.mean(dim=1)
    cx_loss = torch.mean(-torch.log(cx + 1e-5))

    return cx_loss


def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
    return cx


def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W)

    # convert to distance
    dist = 1 - cosine_sim

    return dist


# TODO: Considering avoiding OOM.
def compute_l1_distance(x: torch.Tensor, y: torch.Tensor):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)

    dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    dist = dist.sum(dim=1).abs()
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.)

    return dist


# # TODO: Considering avoiding OOM.
# def compute_l2_distance(x, y):
#     N, C, H, W = x.size()
#     x_vec = x.view(N, C, -1)
#     y_vec = y.view(N, C, -1)
#     x_s = torch.sum(x_vec ** 2, dim=1)
#     y_s = torch.sum(y_vec ** 2, dim=1)

#     A = y_vec.transpose(1, 2) @ x_vec
#     dist = y_s - 2 * A + x_s.transpose(0, 1)
#     dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
#     dist = dist.clamp(min=0.)

#     return dist
def compute_l2_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
    y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)

    A = y_vec.transpose(1, 2) @ x_vec
    # print(x.shape, y_s.shape, A.shape, x_s.shape)
    dist = y_s - 2 * A + x_s.transpose(1, 2)
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.)

    return dist


def compute_meshgrid(shape):
    N, C, H, W = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid


class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Parameters
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 band_width: float = 0.5,
                 loss_type: str = 'cosine',
                 use_vgg: bool = False,
                 vgg_layers: List[str] = ['relu3_4']):

        super(ContextualLoss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert loss_type in LOSS_TYPES, \
            f'select a loss type from {LOSS_TYPES}.'

        self.loss_type = loss_type

        self.band_width = band_width

        if use_vgg:
            self.vgg_model = VGG19()
            self.vgg_layers = vgg_layers
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3, \
                'VGG model takes 3 chennel images.'

            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

            # picking up vgg feature maps
            loss = 0
            vgg19_out_x = self.vgg_model(x)
            vgg19_out_y = self.vgg_model(y)
            for layer in self.vgg_layers:
                x = getattr(vgg19_out_x, layer)
                y = getattr(vgg19_out_y, layer)
                loss += contextual_loss(x, y, band_width=self.band_width, loss_type=self.loss_type)
            return loss / len(self.vgg_layers)

        return contextual_loss(x, y, band_width=self.band_width, loss_type=self.loss_type)


class ContextualBilateralLoss(nn.Module):
    """
    Creates a criterion that measures the contextual bilateral loss.

    Parameters
    ---
    weight_sp : float, optional
        a balancing weight between spatial and feature loss.
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self,
                 weight_sp: float = 0.1,
                 band_width: float = 0.5,
                 loss_type: str = 'cosine',
                 use_vgg: bool = False,
                 vgg_layers: List[str] = ['relu3_4']):

        super(ContextualBilateralLoss, self).__init__()
        self.weight_sp = weight_sp

        assert band_width > 0, 'band_width parameter must be positive.'
        assert loss_type in LOSS_TYPES, \
            f'select a loss type from {LOSS_TYPES}.'

        self.band_width = band_width

        if use_vgg:
            self.vgg_model = VGG19()
            self.vgg_layers = vgg_layers
            self.register_buffer(
                name='vgg_mean',
                tensor=torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False)
            )
            self.register_buffer(
                name='vgg_std',
                tensor=torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False)
            )

    def forward(self, x, y):
        if hasattr(self, 'vgg_model'):
            assert x.shape[1] == 3 and y.shape[1] == 3, \
                'VGG model takes 3 chennel images.'

            # normalization
            x = x.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
            y = y.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

            # picking up vgg feature maps
            loss = 0
            vgg19_out_x = self.vgg_model(x)
            vgg19_out_y = self.vgg_model(y)
            for layer in self.vgg_layers:
                x = getattr(vgg19_out_x, layer)
                y = getattr(vgg19_out_y, layer)
                loss += contextual_bilateral_loss(x, y, band_width=self.band_width, weight_sp=self.weight_sp)

            return loss / len(self.vgg_layers)
        return contextual_bilateral_loss(x, y, band_width=self.band_width, weight_sp=self.weight_sp)
