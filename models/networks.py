import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
from utils import resize_right, interp_methods
from models.blocks import *


VERSION = "faderNet"  # "pix2pixHD"
LD_MULT_BN = False  # unused option, do not change

antialiasing = True
interpolation = interp_methods.cubic

# conv layers of the discriminator


def build_disc_layers(conv_dim=64, n_layers=6, max_dim=512, in_channels=3, activation='relu', normalization='batch', dropout=0):

    # use bias only if we do not use a normalization layer
    bias = normalization != 'batch'
    layers = []
    out_channels = conv_dim

    for i in range(n_layers):
        #print(i, in_channels,out_channels)
        enc_layer = [ConvReluBn(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=bias,
                                padding_mode='reflect'), activation, normalization=normalization if i > 0 else 'none')]
        if dropout > 0:
            enc_layer.append(nn.Dropout(dropout))
        layers.append(nn.Sequential(*enc_layer))
        in_channels = out_channels
        out_channels = min(2*out_channels, max_dim)

    return layers

# conv layers of the encoder


def build_encoder_layers(conv_dim=64, n_layers=6, max_dim=512, im_channels=3, activation='relu', normalization='batch', vgg_like=0, dropout=0, first_conv=False):
    # use bias only if we do not use a normalization layer
    bias = normalization != 'batch'
    if VERSION == "faderNet":
        kernel_sizes = [4, 4, 4, 4, 4, 4, 4, 4, 4]
    else:
        kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
    if first_conv:
        kernel_sizes[0] = 7

    layers = []
    in_channels = im_channels
    out_channels = conv_dim
    for i in range(n_layers):
        #print(i, in_channels,out_channels)
        enc_layer = [ConvReluBn(nn.Conv2d(in_channels, out_channels, kernel_sizes[i], 2 if (i > 0 or not first_conv) else 1, (
            kernel_sizes[i]-1)//2, bias=bias, padding_mode='reflect'), activation, normalization=normalization)]  # PIX2PIX stride 1 in first conv
        if (i >= n_layers-1-vgg_like and i < n_layers-1):
            enc_layer += [ConvReluBn(nn.Conv2d(out_channels, out_channels, 3, 1,
                                     1, bias=bias, padding_mode='reflect'), activation, normalization)]
            #enc_layer += [ConvReluBn(nn.Conv2d(out_channels, out_channels, 3, 1, 1,bias=bias),activation,normalization)]
        if dropout > 0:
            enc_layer.append(nn.Dropout(dropout))
        layers.append(nn.Sequential(*enc_layer))
        in_channels = out_channels
        out_channels = min(2*out_channels, max_dim)
    return layers


# encoder : conv layers + bottleneck
class Encoder(nn.Module):
    def __init__(self, conv_dim, n_layers, max_dim, im_channels, vgg_like, activation='relu', normalization='batch', first_conv=False, n_bottlenecks=2):

        super(Encoder, self).__init__()
        # use bias only if we do not use a normalization layer
        bias = normalization != 'batch'
        enc_layers = build_encoder_layers(conv_dim, n_layers, max_dim, im_channels, normalization=normalization,
                                          activation=activation, vgg_like=vgg_like, first_conv=first_conv)
        self.encoder = nn.ModuleList(enc_layers)
        b_dim = min(max_dim, conv_dim * 2 ** (n_layers-1))
        self.bottleneck = nn.ModuleList([ResidualBlock(
            b_dim, b_dim, activation, normalization, bias=bias) for i in range(n_bottlenecks)])

    # return [encodings,bneck]
    def encode(self, x):
        # Encoder
        x_encoder = []
        for block in self.encoder:
            x = block(x)
            x_encoder.append(x)

        bn = []
        # Bottleneck
        for block in self.bottleneck:
            x = block(x)
            bn.append(x)
        return x_encoder, x, bn

# trial to preprocess the attribute (a few FC layers), not used in the end


def attribute_pre_treat(attr_dim, first_dim, max_dim, n_layers):
    # linear features for attributes
    layers = []
    in_channels = attr_dim
    out_channels = first_dim
    for i in range(n_layers):
        layers.append(nn.Sequential(nn.Linear(in_channels, out_channels),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        in_channels = out_channels
        out_channels = min(2*out_channels, max_dim)
    return nn.Sequential(*layers)

# used for the attribute: repeat spacially the attribute a and concat to feature


def reshape_and_concat(feat, a):
    a = a.unsqueeze(-1).unsqueeze(-1)
    attr = a.repeat((1, 1, feat.size(2), feat.size(3)))
    return torch.cat([feat, attr], dim=1)

# decoder conv layers


def build_decoder_layers(conv_dim=64, n_layers=6, max_dim=512, im_channels=3, skip_connections=0, attr_dim=0, n_attr_deconv=0, vgg_like=0, n_branches=1, activation='leaky_relu', normalization='batch', additional_channels=[], first_conv=False):  # NEW leaky_relu by default
    bias = normalization != 'batch'
    decoder = nn.ModuleList()
    # PIX2PIX do no put the very last intermediate convolutions (one less stride)
    shift = 0 if not first_conv else 1
    for i in reversed(range(shift, n_layers)):
        # size of inputs/outputs
        dec_out = int(min(max_dim, conv_dim * 2 ** (i-1)))
        dec_in = min(max_dim, conv_dim * 2 ** (i))
        # corresponding encoding size (for skip connections)
        enc_size = min(max_dim, conv_dim * 2 ** (i))

        if i == n_layers-1:
            dec_in = enc_size * n_branches
        if i >= n_layers - n_attr_deconv:
            dec_in = dec_in + attr_dim  # concatenate attribute
        # skip connection: n_branches-1 or 1 feature map
        if i >= n_layers - 1 - skip_connections and i != n_layers-1:
            dec_in = dec_in + max(1, n_branches-1)*enc_size
        if (i == shift and VERSION == "faderNet"):
            dec_out = conv_dim // 4  # last conv is of dim dim / 4
        if (i-shift < len(additional_channels)):
            # PIX2PIX there is one layer less than n_layers
            dec_in += additional_channels[i-shift]
        # print(i,dec_in)

        dec_layer = [ConvReluBn(nn.Conv2d(dec_in, dec_out, 3, 1, 1, bias=bias,
                                padding_mode='reflect'), activation=activation, normalization=normalization)]
        if (vgg_like > 0 and i >= n_layers - vgg_like) or (i == shift and VERSION == "faderNet" and len(additional_channels) > 0):
            dec_layer += [ConvReluBn(nn.Conv2d(dec_out, dec_out, 3, 1, 1,
                                     bias=bias, padding_mode='reflect'), activation, normalization)]
        decoder.append(nn.Sequential(*dec_layer))

    last_kernel = 3 if VERSION == "faderNet" else 7
    # PIX2PIX last conv has kernel 7, padding 3
    last_conv = nn.ConvTranspose2d(
        dec_out, im_channels, last_kernel, 1, last_kernel//2, bias=True)
    return decoder, last_conv


# a simple unet (encoder decoder)
class Unet(nn.Module):
    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, skip_connections=2, vgg_like=0, normalization='instance', first_conv=False, n_bottlenecks=2, img_size=128, batch_size=32, device='cpu'):

        super(Unet, self).__init__()
        self.n_layers = n_layers
        self.skip_connections = min(skip_connections, n_layers - 1)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.resize_layers = []
        for n in range(n_layers):
            self.resize_layers.append(resize_right.ResizeLayer([batch_size, min(max_dim, conv_dim * 2 ** (n)), img_size/(2 ** (
                n+1)), img_size/(2 ** (n+1))], scale_factors=2, device=device, interp_method=interpolation, antialiasing=antialiasing))
        # build encoder
        self.encoder = Encoder(conv_dim, n_layers, max_dim, im_channels, vgg_like,
                               normalization=normalization, first_conv=first_conv, n_bottlenecks=n_bottlenecks)
        # build decoder
        self.decoder, self.last_conv = build_decoder_layers(
            conv_dim, n_layers, max_dim, 3, skip_connections=skip_connections, vgg_like=vgg_like, normalization=normalization)

    # adding the skip connection if needed
    def add_skip_connection(self, i, out, encodings):
        if 0 < i <= self.skip_connections:
            out = torch.cat([out, encodings[-(i+1)]], dim=1)
        return out

    # return [encodings,bneck]
    def encode(self, x):
        return self.encoder.encode(x)

    def decode(self, bneck, encodings):
        out = bneck
        for i, dec_layer in enumerate(self.decoder):
            out = self.add_skip_connection(i, out, encodings)
            #out = dec_layer(self.up(out))
            #out = dec_layer(resize_right.resize(out, scale_factors=2))
            out = dec_layer(self.resize_layers[self.n_layers-i-1])
        x = self.last_conv(out)
        x = torch.tanh(x)
        x = x / torch.sqrt((x**2).sum(dim=1, keepdims=True))
        return x

    def forward(self, x):
        # propagate encoder layers
        encodings, z, _ = self.encode(x)
        return self.decode(z, encodings)


# G1 without normals (faderNet)
class FaderNetGenerator(Unet):
    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, skip_connections=2, vgg_like=0, attr_dim=1, n_attr_deconv=1, normalization='instance', first_conv=False, n_bottlenecks=2, img_size=128, batch_size=32, device='cpu'):

        super(FaderNetGenerator, self).__init__(conv_dim, n_layers, max_dim, im_channels,
                                                skip_connections, vgg_like, normalization, first_conv, n_bottlenecks, img_size, batch_size, device=device)
        self.attr_dim = attr_dim
        self.n_attr_deconv = n_attr_deconv

        # change decoder : get attribute as input
        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim, 3, skip_connections=skip_connections,
                                                            vgg_like=vgg_like, attr_dim=attr_dim, n_attr_deconv=n_attr_deconv, normalization=normalization, first_conv=first_conv)

        # bottlenecks
        bias = normalization != 'batch'
        b_dim = min(max_dim, conv_dim * 2 ** (n_layers-1))
        n_bottlenecks_dec = 0  # bottlenecks that treat the attribute
        self.bottleneck = nn.ModuleList([ResidualBlock(
            b_dim+attr_dim, b_dim, 'relu', normalization, bias=bias) for i in range(n_bottlenecks_dec)])

    # adding the attribute if needed
    def add_attribute(self, i, out, a):
        if i < self.n_attr_deconv:
            out = reshape_and_concat(out, a)
        return out

    # go through decoder's bottleneck  (with attribute)
    def decoder_bottlenck(self, bneck, a):
        for block in self.bottleneck:
            bneck = block(bneck, a)
        return bneck

    def decode(self, a, bneck, encodings):
        bneck = self.decoder_bottlenck(bneck, a)
        out = bneck
        for i, dec_layer in enumerate(self.decoder):
            out = self.add_attribute(i, out, a)
            out = self.add_skip_connection(i, out, encodings)
            out = dec_layer(self.up(out))
        x = self.last_conv(out)
        x = torch.tanh(x)

        return x

    def forward(self, x, a):
        # propagate encoder layers
        encodings, z, _ = self.encode(x)
        return self.decode(a, z, encodings)

# G1 (faderNet with normals)


class FaderNetGeneratorWithNormals(FaderNetGenerator):

    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, skip_connections=2, vgg_like=0, attr_dim=1, n_attr_deconv=1, n_concat_normals=1, normalization='instance', first_conv=False, n_bottlenecks=2, img_size=128, batch_size=32, device='cpu'):
        super(FaderNetGeneratorWithNormals, self).__init__(conv_dim, n_layers, max_dim, im_channels, skip_connections,
                                                           vgg_like, attr_dim, n_attr_deconv, normalization, first_conv, n_bottlenecks, img_size, batch_size, device=device)

        self.n_concat_normals = n_concat_normals
        self.first_conv = first_conv
        self.pretreat_attr = False

        # change decoder : get normal as input
        dim_attr_treat = 32
        additional_channels = [3 for i in range(self.n_concat_normals)]
        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim, 3, skip_connections=skip_connections, vgg_like=0 if VERSION == "faderNet" else 3,
                                                            attr_dim=attr_dim if not self.pretreat_attr else dim_attr_treat, n_attr_deconv=n_attr_deconv, additional_channels=additional_channels, normalization=normalization, first_conv=first_conv)  # NEW vgg_like=3
        self.attr_FC = attribute_pre_treat(
            attr_dim, dim_attr_treat, dim_attr_treat, 2)

        self.resize_normals = []
        for n in range(n_concat_normals-1):
            self.resize_normals.append(resize_right.ResizeLayer([batch_size, 3, img_size, img_size], scale_factors=0.5**(
                n+1), device=device, interp_method=interpolation, antialiasing=antialiasing))

    def prepare_pyramid(self, map, n_levels):
        map_pyramid = [map]

        for i in range(n_levels-1):
            #map_pyramid.insert(0,nn.functional.interpolate(map_pyramid[0], mode='bilinear', align_corners=False, scale_factor=0.5))
            map_pyramid.insert(0, self.resize_normals[i](map))
            #map_pyramid.insert(0,resize_right.resize(map_pyramid[0], scale_factors=0.5, interp_method=interp_methods.cubic,antialiasing=True))
        return map_pyramid

    # adding the normal map at the right scale if needed
    def add_multiscale_map(self, i, out, map_pyramid, n_levels):
        # PIX2PIX there is one layer less than n_layers
        shift = 0 if not self.first_conv else 1
        rank = i-(self.n_layers-shift-n_levels)
        if rank >= 0 and rank < len(map_pyramid):
            out = (torch.cat([out, map_pyramid[rank]], dim=1))
        return out

    def decode(self, a, bneck, normals, encodings):
        x, features = self.decode_with_features(a, bneck, normals, encodings)
        return x

    # decode and return all the feature maps
    def decode_with_features(self, a, bneck, normals, encodings):
        features = []
        # prepare attr
        if self.pretreat_attr:
            a = self.attr_FC(a)
        # prepare normals
        normal_pyramid = self.prepare_pyramid(normals, self.n_concat_normals)
        # go through net
        bneck = self.decoder_bottlenck(bneck, a)
        out = bneck
        for i, dec_layer in enumerate(self.decoder):
            out = self.add_skip_connection(i, out, encodings)
            #out = self.up(out)
            out = resize_right.resize(out, scale_factors=2)
            out = self.add_attribute(i, out, a)
            out = self.add_multiscale_map(
                i, out, normal_pyramid, self.n_concat_normals)
            out = dec_layer(out)
            features.append(out)
        x = self.last_conv(out)
        x = torch.tanh(x)
        return x, features

    def forward(self, x, a, normals):
        # propagate encoder layers
        encodings, z, _ = self.encode(x)
        return self.decode(a, z, normals, encodings)


# generator G2
class FaderNetGeneratorWithNormals2Steps(FaderNetGeneratorWithNormals):

    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, skip_connections=2, vgg_like=0, attr_dim=1, n_attr_deconv=1, n_concat_normals=1, normalization='instance', first_conv=False, n_bottlenecks=2, all_feat=True, img_size=128, batch_size=32, device='cpu'):
        super(FaderNetGeneratorWithNormals2Steps, self).__init__(conv_dim, n_layers, max_dim, im_channels, skip_connections,
                                                                 vgg_like, attr_dim, n_attr_deconv, n_concat_normals, normalization, first_conv, n_bottlenecks, img_size, batch_size, device=device)
        self.all_feat = all_feat
        self.pretreat_attr = False
        self.img_size = img_size

        feat_channels = [8, 32, 64, 128, 256]
        if self.img_size > 128:
            feat_channels.insert(0, 0)
        additional_channels = [3+feat_channels[i if all_feat else 0]
                               for i in range(self.n_concat_normals)]

        dim_attr_treat = 16
        self.attr_FC = attribute_pre_treat(
            attr_dim, dim_attr_treat, dim_attr_treat, 2)

        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim, 3, skip_connections=skip_connections, vgg_like=vgg_like, attr_dim=attr_dim if not self.pretreat_attr else dim_attr_treat,
                                                            n_attr_deconv=n_attr_deconv, additional_channels=additional_channels, normalization=normalization, first_conv=first_conv)

    def decode(self, a, bneck, normals, fadernet_output, encodings):
        if self.pretreat_attr:
            a = self.attr_FC(a)
        # prepapre illum and normals
        if self.all_feat:
            fadernet_pyramid = fadernet_output[-self.n_concat_normals:]
        else:
            fadernet_pyramid = self.prepare_pyramid(
                fadernet_output[-1], self.n_concat_normals)
        normal_pyramid = self.prepare_pyramid(normals, self.n_concat_normals)
        # go through decoder
        bneck = self.decoder_bottlenck(bneck, a)
        out = bneck
        for i, dec_layer in enumerate(self.decoder):
            out = self.add_skip_connection(i, out, encodings)
            # out=self.up(out)
            out = resize_right.resize(out, scale_factors=2)
            out = self.add_attribute(i, out, a)
            out = self.add_multiscale_map(
                i, out, normal_pyramid, self.n_concat_normals)
            out = self.add_multiscale_map(
                i+(1 if self.img_size > 128 else 0), out, fadernet_pyramid, self.n_concat_normals)
            out = dec_layer(out)
        x = self.last_conv(out)
        x = torch.tanh(x)

        return x

    def forward(self, x, a, normals):
        # propagate encoder layers
        encodings, z = self.encode(x)
        return self.decode(a, z, normals, encodings)


# trial of G2 + illumination map, not used
class FaderNetGeneratorWithNormalsAndIllum(FaderNetGeneratorWithNormals):
    def __init__(self, conv_dim=64, n_layers=5, max_dim=1024, im_channels=3, skip_connections=2, vgg_like=0, attr_dim=1, n_attr_deconv=1, n_concat_normals=1, n_concat_illum=1, normalization='instance'):
        super(FaderNetGeneratorWithNormalsAndIllum, self).__init__(conv_dim, n_layers, max_dim,
                                                                   im_channels, skip_connections, vgg_like, attr_dim, n_attr_deconv, n_concat_normals, normalization)

        self.n_concat_illum = n_concat_illum
        # change decoder : add illum as input
        # TODO separate illum from normals
        additional_channels = [3+6 for i in range(self.n_concat_normals)]
        self.decoder, self.last_conv = build_decoder_layers(conv_dim, n_layers, max_dim, 3, skip_connections=skip_connections, vgg_like=vgg_like,
                                                            attr_dim=attr_dim, n_attr_deconv=n_attr_deconv, additional_channels=additional_channels, normalization=normalization)

        bias = norm == 'none'
        # series of convolutions for the illum
        enc_layer = [ConvReluBn(nn.Conv2d(self.attr_dim+3, 6, 3, 1, 1, bias=bias), activation, normalization=normalization),
                     ConvReluBn(nn.Conv2d(6, 6, 3, 1, 1, bias=bias),
                                activation, normalization=normalization),
                     ConvReluBn(nn.Conv2d(6, 6, 3, 1, 1, bias=bias), activation, normalization=normalization)]
        self.illum_conv = (nn.Sequential(*enc_layer))

    def decode(self, a, bneck, normals, illum, encodings):
        # prepapre illum and normals
        illum = self.illum_conv(reshape_and_concat(illum, a), dim=1)
        illum_pyramid = self.prepare_pyramid(illum, self.n_concat_illum)
        normal_pyramid = self.prepare_pyramid(normals, self.n_concat_normals)
        # go through decoder
        bneck = self.decoder_bottlenck(bneck, a)
        out = bneck
        for i, dec_layer in enumerate(self.decoder):
            out = self.add_attribute(i, out, a)
            out = self.add_skip_connection(i, out, encodings)
            out = self.up(out)
            out = self.add_multiscale_map(
                i, out, normal_pyramid, self.n_concat_normals)
            out = self.add_multiscale_map(
                i, out, illum_pyramid, self.n_concat_illum)
            out = dec_layer(out)
        x = self.last_conv(out)
        x = torch.tanh(x)

        return x

    def forward(self, x, a, normals):
        # propagate encoder layers
        encodings, z = self.encode(x)
        return self.decode(a, z, normals, encodings)


def FC_layers(in_dim, fc_dim, out_dim, tanh):
    if VERSION == "faderNet":
        layers = [nn.Linear(in_dim, fc_dim),
                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                  nn.Linear(fc_dim, out_dim)]
    else:
        layers = [nn.Linear(in_dim, out_dim)]  # NEW only one FC
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# LD
class Latent_Discriminator(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, im_channels=3, conv_dim=64, fc_dim=1024, n_layers=5, skip_connections=2, vgg_like=0, normalization='instance', first_conv=False):
        super(Latent_Discriminator, self).__init__()
        layers = []
        self.n_bnecks = 3
        n_dis_layers = int(np.log2(image_size))
        layers = build_encoder_layers(conv_dim, n_dis_layers, max_dim, im_channels,
                                      normalization=normalization, activation='leaky_relu', dropout=0.3, first_conv=first_conv)
        # NEW change first conv to get 3 times bigger input
        if LD_MULT_BN:
            layers[n_layers-skip_connections][0].conv = nn.Conv2d(layers[n_layers-skip_connections][0].conv.in_channels*self.n_bnecks, layers[n_layers-skip_connections][0].conv.out_channels,
                                                                  layers[n_layers-skip_connections][0].conv.kernel_size, layers[n_layers-skip_connections][0].conv.stride, 1, bias=normalization != 'batch', padding_mode='reflect')

        self.conv = nn.Sequential(*layers[n_layers-skip_connections:])
        self.pool = nn.AvgPool2d(1 if not first_conv else 2)
        out_conv = min(max_dim, conv_dim * 2 ** (n_dis_layers - 1))
        self.fc_att = FC_layers(out_conv, fc_dim, attr_dim, True)

    def forward(self, x, bn_list):
        if LD_MULT_BN:
            x = torch.cat(bn_list[-self.n_bnecks:], dim=1)
        y = self.conv(x)
        y = self.pool(y)
        y = y.view(y.size()[0], -1)
        logit_att = self.fc_att(y)
        return logit_att


# classical GAN discriminator
class Discriminator(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, im_channels=3, conv_dim=64, fc_dim=1024, n_layers=5, normalization='instance'):
        super(Discriminator, self).__init__()
        layers = []
        layers = build_disc_layers(conv_dim, n_layers, max_dim, im_channels,
                                   normalization=normalization, activation='leaky_relu')
        self.conv = nn.Sequential(*layers)

        c_dim = min(max_dim, conv_dim * 2 ** (n_layers-1))
        # size of kernel: 1,3 or 4
        self.last_conv = nn.Conv2d(c_dim, 1, 4, 1, 1)

    def forward(self, x):
        y = self.conv(x)
        logit_real = self.last_conv(y)
        return logit_real

# discriminator that also takes as input the attribute (answer false if the pair is not good as 2nd output)


class DiscriminatorWithMatchingAttr(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, im_channels=3, conv_dim=64, fc_dim=1024, n_layers=5, normalization='instance'):
        super(DiscriminatorWithMatchingAttr, self).__init__()
        # convolutions for image
        layers = build_disc_layers(conv_dim, n_layers, max_dim, im_channels,
                                   normalization=normalization, activation='leaky_relu')
        self.conv_img = nn.Sequential(*layers)

        c_dim = min(max_dim, conv_dim * 2 ** (n_layers-1))
        self.last_img_conv = nn.Conv2d(
            c_dim, 1, 4, 1, 1)  # size of kernel: 1,3 or 4

        # linear features for attributes
        n_layers_attr = 2
        self.linear_attr = attribute_pre_treat(
            attr_dim, conv_dim, max_dim, n_layers_attr)
        # or attr_dim if nor using linear_attr
        n_feat_attr = min(max_dim, conv_dim * 2 ** (n_layers_attr-1))

        activation = 'leaky_relu'
        bias = normalization != 'batch'
        c_dim = min(max_dim, conv_dim * 2 ** (n_layers-1))
        self.last_conv = nn.Sequential(ConvReluBn(nn.Conv2d(c_dim+n_feat_attr, c_dim, 1, 1, 0, bias=bias), activation, normalization),
                                       nn.Conv2d(c_dim, 1, 4, 1, 1))  # size of kernel: 1,3 or 4

    def forward(self, x, attr):
        img_feat = self.conv_img(x)
        attr_feat = self.linear_attr(attr)
        logit_matching = self.last_conv(
            reshape_and_concat(img_feat, attr_feat))
        logit_real = self.last_img_conv(img_feat)
        return logit_real, logit_matching

# discriminator that also tries to guess the attribute


class DiscriminatorWithClassifAttr(nn.Module):
    def __init__(self, image_size=128, max_dim=512, attr_dim=10, im_channels=3, conv_dim=64, fc_dim=1024, n_layers=5, normalization='instance'):
        super(DiscriminatorWithClassifAttr, self).__init__()
        # convolutions for image
        layers = build_disc_layers(conv_dim, n_layers, max_dim, im_channels,
                                   normalization=normalization, activation='leaky_relu')
        self.conv_img = nn.Sequential(*layers)

        c_dim = min(max_dim, conv_dim * 2 ** (n_layers-1))
        self.last_img_conv = nn.Conv2d(
            c_dim, 1, 4, 1, 1)  # size of kernel: 1,3 or 4

        self.pool = nn.AvgPool2d(int(image_size/(2**n_layers)))
        self.fc_att = FC_layers(c_dim, fc_dim, attr_dim, True)

    def forward(self, x):
        img_feat = self.conv_img(x)
        y = self.pool(img_feat)
        y = y.view(y.size()[0], -1)
        logit_att = self.fc_att(y)
        logit_real = self.last_img_conv(img_feat)
        return logit_real, logit_att


if __name__ == '__main__':
    gen = Generator(5, conv_dim=32, n_layers=6,
                    skip_connections=0, max_dim=512, vgg_like=True)

    print(gen)
    #summary(gen, [(3, 128, 128), (5,)], device='cpu')

    # dis = Discriminator(image_size=128, max_dim=512, attr_dim=5,conv_dim=32,n_layers=7)
    # print(dis)
    # #summary(dis, (3, 128, 128), device='cpu')

    # dis = Latent_Discriminator(image_size=128, max_dim=512, attr_dim=5,conv_dim=32,n_layers=6,skip_connections=2)
    # print(dis)
