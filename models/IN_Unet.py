""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F
import torch
import torch.nn as nn
# from tsnecuda import TSNE
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
# import numpy as np
# from sklearn.cluster import KMeans

class InstanceNormalization_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, is_normalize:bool, bilinear=False, is_cls:bool=True,instance_branch:bool=False,pad_mode:bool=True):
        super(InstanceNormalization_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.is_cls = is_cls
        self.instance_branch = instance_branch
        self.pad_mode = pad_mode
        # self.is_styleLoss = True
        self.is_normalize = is_normalize
        self.dummy_param = nn.Parameter(torch.empty(0)) #check device is cuda or cpu

        self.mse_loss = nn.MSELoss() # calculate style loss

        self.encoder = Encoder(n_channels, n_layers=4,is_normalize=is_normalize)
        self.decoder = Decoder(n_classes=n_classes,n_layers=4,is_normalize=is_normalize,pad_mode=self.pad_mode,bilinear=bilinear,is_cls=self.is_cls)

    def forward(self, x):
        code = self.encoder(x)
        source_fs = self.encoder.features
        logits = self.decoder(code, source_fs)

        if not self.pad_mode:
            diffY = torch.tensor([x.shape[2] - logits.size()[2]])
            diffX = torch.tensor([x.shape[3] - logits.size()[3]])
            logits = F.pad(logits, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],mode="replicate")

        return logits
    
    def freeze_encoder(self, is_freeze:bool):
        for name, p in self.encoder.named_parameters():
            p.requires_grad = not is_freeze
            # print(f"{name}: {p.requires_grad}")
    
class Encoder(nn.Module):
    def __init__(self, n_channels, n_layers, is_normalize:bool):
        super(Encoder, self).__init__()
        self.inc = DoubleConv(n_channels, 64, is_normalize=is_normalize)
        self.n_layers = n_layers
        self.layers = nn.ModuleList([Down(64*(2**i), 64*(2**(i+1)),is_normalize=is_normalize) for i in range(n_layers)])
        self.features = []

    def forward(self, x):
        self.features = [] # initial encoder_fs at every batch
        x = self.inc(x)
        self.features.append(x)
        for i,layer in enumerate(self.layers):
            x = layer(x)
            if i+1 != self.n_layers:
                self.features.append(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, n_classes, n_layers, is_normalize:bool, pad_mode:bool=True, bilinear:bool=False, is_cls:bool=True):
        super(Decoder, self).__init__()
        self.bilinear = bilinear
        self.n_layers = n_layers
        self.is_cls = is_cls
        self.pad_mode = pad_mode
        self.layers = nn.ModuleList([Up(64*(2**(i+1)),64*(2**i),is_normalize,bilinear,pad_fmap=pad_mode) for i in range(n_layers-1,-1,-1)])
        self.features = []
        if is_cls:
            self.outc = OutConv(64, n_classes)
        
    def forward(self, x, encoder_fs):
        assert len(encoder_fs) == self.n_layers
        self.features = []
        for i,layer in enumerate(self.layers):
            x = layer(x,encoder_fs[-(i+1)])
            self.features.append(x)
        if self.is_cls:
            x = self.outc(x)

        return x

class DoubleConv(nn.Module):
    """(convolution => [IN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, is_normalize:bool=True):
        super().__init__()
        self.is_normalize = is_normalize
        if is_normalize:
            double_conv = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)]
        else:
            double_conv = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)]
            
        self.double_conv = nn.Sequential(*double_conv)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, is_normalize:bool):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, is_normalize=is_normalize)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, is_normalize:bool, bilinear=True,pad_fmap:bool=True):
        super().__init__()
        self.pad_fmap = pad_fmap
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, is_normalize=is_normalize)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BCHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        if self.pad_fmap:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        else:
            # center crop to x2
            x2_orig_y, x2_orig_x = x2.shape[2], x2.shape[3]
            x2 = x2[:,:, (diffY//2):x2_orig_y-(diffY - diffY // 2), (diffX//2):x2_orig_x-(diffX - diffX // 2)]
    
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
def cross_entropy_2d(predict, target):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, 1, h, w) or (n, h, w)
    """
    target = target.long()
    if target.dim() == 4:
        target = target[:,0,:,:]
    assert not target.requires_grad
    assert predict.dim() == 4
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    # print(f" target_mask shape: {target_mask.shape}") #(B,H,W)
    # print(target_mask)
    target = target[target_mask]
    # print(f" label shape: {target.shape}")
    if not target.data.dim():
        return torch.zeros(1)
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous() # (n,c,h,w) -> (n,h,w,c)
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, size_average=True)
    return loss

def calc_mean_std(feat, eps=1e-7):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    
if __name__ == '__main__':
    net = InstanceNormalization_UNet(n_channels=1, n_classes=2,is_normalize=True)
    net.freeze_encoder(is_freeze=True)
    
