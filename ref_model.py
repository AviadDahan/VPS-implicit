from models.implicit_neural_networks import IMLP
from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def norm_z(z):
    return z / z.norm(dim=1).unsqueeze(dim=1)


class RefineModel(nn.Module):
    def __init__(self, args, output_dim=1, use_tanh=False, apply_softmax=False):
        super(RefineModel, self).__init__()
        self.model_alpha = IMLP(
            input_dim=3,
            output_dim=output_dim,
            hidden_dim=int(args['number_of_channels_alpha']),
            use_positional=True,
            positional_dim=int(args['positional_encoding_num_alpha']),
            num_layers=int(args['number_of_layers_alpha']),
            num_of_ex=1,
            skip_layers=[4, 6],
            use_tanh=use_tanh,
            apply_softmax=apply_softmax)

    def forward(self, pos):
        zeros = torch.tensor(0).repeat(pos.shape[0]).cuda()
        # out = torch.sigmoid(self.model_alpha(pos, zeros))
        out = self.model_alpha(pos, zeros)
        return out


class ImageEncoderResnet(nn.Module):
    def __init__(self, args):
        super(ImageEncoderResnet, self).__init__()
        res = int(args['order'])
        if res == 18:
            emb_size = 512
        elif res == 34:
            emb_size = 512
        elif res == 50:
            emb_size = 2048
        elif res == 101:
            emb_size = 2048
        else:
            emb_size = 512
        self.E = ResNet(res=res)
        self.fc1 = nn.Linear(emb_size, 256)

    def forward(self, x):
        x = self.E(x)
        x = self.fc1(x)
        return x


class ResNet(nn.Module):
    def __init__(self, res=50, is_grad=True):
        super(ResNet, self).__init__()
        if res == 18:
            self.Org_model = models.resnet18(pretrained=True)
        if res == 34:
            self.Org_model = models.resnet34(pretrained=True)
        if res == 50:
            self.Org_model = models.resnet50(pretrained=True)
        if res == 101:
            self.Org_model = models.resnet101(pretrained=True)
        for param in self.Org_model.parameters():
            param.requires_grad = is_grad

    def forward(self, x_in):
        x = self.Org_model.conv1(x_in)
        x = self.Org_model.bn1(x)
        x = self.Org_model.relu(x)
        x = self.Org_model.maxpool(x)
        x = self.Org_model.layer1(x)
        x = self.Org_model.layer2(x)
        x = self.Org_model.layer3(x)
        x = self.Org_model.layer4(x)
        x = self.Org_model.avgpool(x).squeeze(dim=2).squeeze(dim=2)
        return x


class NetF(nn.Module):
    '''
    An encoder-style network which outputs an embedding space from which each layer in NetG will predict its weights.
    NOTE: In the usual defenition of f and g, the output of f is the weights for g. Implementation-wise it is more
    convinient for f to output an embedding space while each layer in g will predict its weights from the embedding space given by f.
    '''
    def __init__(self, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, emb_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        emb = self.fc2(x)
        return emb


class MetaConv(nn.Module):
    '''
    A "meta" convolution layer.
    NOTE: Bias is not yet implemented
    Inputs:
        n_filt_in: num of filters entering the layer
        n_filt_out: num of filters to output
        kernel_size: kernel size
        g_latent_dim: intermidiate dimension for the FC layers predicting the weights from f's embedding
        emb_dim: the embedding dimension given by f
    '''

    def __init__(self, n_filt_in, n_filt_out, kernel_size, g_latent_dim, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.act_f = nn.ReLU()
        self.w_dim = (n_filt_out, n_filt_in, kernel_size, kernel_size)

        self.stride, self.padding, self.output_padding, self.groups, self.dilation = ((1, 1), (0, 0), (0, 0), 1, (1, 1))
        self.w1 = nn.Linear(self.emb_dim, g_latent_dim)
        self.w2 = nn.Linear(g_latent_dim, np.prod(self.w_dim))

    def forward(self, emb, x):
        '''
        Recieves the embedding given by f, and predicts the weighs for this layer.
        NOTE: each sample in the batch x needs to be convolved with different weights.
        the trick is to reshape x into (1, bs * c, h, w) and stack the weights to be (bs * out_channels, in_channels, k, k),
        and then do a group convolution, and reshape the output to the original dimensions of x.
        '''
        weights = self.get_weights(emb)

        # reshaping for using group covolution
        bs, c, h, w = x.shape
        xtag = x.contiguous().view(1, bs * c, h, w)
        bs, out_channels, in_channels, k, k = weights.shape
        weights = weights.view(bs * out_channels, in_channels, k, k)

        # group convolution
        out = F.conv2d(xtag, weight=weights, bias=None, stride=self.stride, dilation=self.dilation, groups=bs,
                       padding=self.padding)

        # reshape back to normal batch size
        out = out.view(bs, out_channels, out.shape[-2], out.shape[-1])
        return out

    def get_weights(self, emb):
        '''
        predicts the weights for this layers given the embedding output by f.
        '''
        w = self.act_f(self.w1(emb))
        return self.w2(w).view(-1, self.w_dim[0], self.w_dim[1], self.w_dim[2], self.w_dim[3])


class MetaLinear(nn.Module):
    '''
    A 'meta' fully-connected layer.
    inputs:
        nin: number of input features
        nout: number of output features
        emb_dim: the embedding dimension given by f
    '''
    def __init__(self, nin, nout, emb_dim):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.lkrelu = nn.LeakyReLU()

        self.w1 = torch.nn.Linear(emb_dim, 2*nin*nout)
        self.w2 = torch.nn.Linear(2*nin*nout, nin*nout)

        self.b1 = torch.nn.Linear(emb_dim, 2*nout)
        self.b2 = torch.nn.Linear(2*nout, nout)

    def forward(self, emb, x):
        '''
        Recieves the embedding given by f and predicts this g layer's weights.
        '''
        w, b = self.get_weights(emb)
        x = x.reshape(x.shape[0], 1, x.shape[-1])
        # y = torch.bmm(x, w) + b
        return (x@w.squeeze(dim=0)).squeeze(dim=2)

    def get_weights(self, emb):
        '''
        predicts the weights for this layers given the embedding output by f.
        '''
        w = self.lkrelu(self.w1(emb))
        w = self.w2(w)
        w = w.view(emb.shape[0], self.nin, self.nout)

        b = self.lkrelu(self.b1(emb))
        b = self.b2(b)
        b = b.unsqueeze(1)
        return w, b


class NetG(nn.Module):
    '''
    Defines the network g by multiple meta conv / meta fc layers.
    Each layer in g predicts its own weights from the embedding given by f.
    inputs:
        g_latent_dim: intermidiate dimension for the FC layers predicting the weights from f's embedding
        emb_dim: the dimension of the embedding given by f
    '''

    def __init__(self, emb_dim, input_dim, positional_dim):
        super().__init__()
        encoding_dimensions = 2 * input_dim * positional_dim
        self.b = torch.tensor([(2 ** j) * np.pi for j in range(positional_dim)], requires_grad=False).cuda()
        self.fc1 = nn.Linear(encoding_dimensions, 256)
        self.fc2 = nn.Linear(256, 256)
        self.meta_fc3 = MetaLinear(256, 1, emb_dim)

    def forward(self, emb, x):
        x = positionalEncoding_vec(x, self.b)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.meta_fc3(emb.mean(dim=0).unsqueeze(dim=0), x)
        return F.tanh(x)


def positionalEncoding_vec(in_tensor, b):
    proj = torch.einsum('ij, k -> ijk', in_tensor, b)  # shape (batch, in_tensor.size(1), freqNum)
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  # shape (batch, 2*in_tensor.size(1), freqNum)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output