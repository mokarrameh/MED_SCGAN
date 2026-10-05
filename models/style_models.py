# Style-Enhanced CycleGAN networks: mapping network, AdaIN, and the style-modulated ResNet
# and U-Net generators. The StyleGAN components (PixelNorm, the equalized-learning-rate layers
# WSLinear and WSConv2d, MappingNetwork, AdaIN) follow Karras et al., "A Style-Based Generator
# Architecture for Generative Adversarial Networks", CVPR 2019.
# init_weights/init_net, define_D, NLayerDiscriminator and the recursive U-Net structure are
# adapted from models/networks.py of pytorch-CycleGAN-and-pix2pix
# (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix),
# Copyright (c) 2017, Jun-Yan Zhu and Taesung Park. BSD license; see NOTICE.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.models as models
from torchvision.transforms import Normalize

def init_weights(net, init_type='normal', init_gain=0.02, skip_types=(None,)):
    """Initialize network weights, skipping layers of types in skip_types.

    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)   -- scaling factor for normal, xavier, and orthogonal
        skip_types (tuple)  -- types of modules to skip during initialization
    """
    def init_func(m):
        if isinstance(m, skip_types):
            return  # skip this module

        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print(f'Initialize network with {init_type}, skipping {skip_types}')
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network, skipping MappingNetwork submodules.

    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- initialization method
        init_gain (float)  -- scaling factor for initialization
        gpu_ids (list)     -- list of GPU ids to use

    Returns:
        Initialized network
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    # Skip MappingNetwork layers
    init_weights(net, init_type, init_gain=init_gain, skip_types=(MappingNetwork,WSLinear,WSConv2d))
    return net

class WSLinear(nn.Module):
    """
    Weight-Scaled Linear Layer (WSLinear).

    This layer implements the "Equalized Learning Rate" trick introduced in StyleGAN.
    Instead of relying on weight initialization alone to control the variance of
    activations, the inputs are dynamically scaled by sqrt(2 / fan_in) at runtime.
    This stabilizes training and ensures consistent signal magnitudes.

    Parameters
    ----------
    in_features : int
        Size of each input sample (fan_in).
    out_features : int
        Size of each output sample (fan_out).

    Attributes
    ----------
    linear : nn.Linear
        Underlying linear layer without bias (weights only).
    scale : float
        Scaling factor applied to the input, computed as sqrt(2 / in_features).
    bias : nn.Parameter
        Bias term applied after the linear transformation.

    Notes
    -----
    - The weights are initialized from a standard normal distribution (N(0,1)).
    - The bias is initialized to zero.
    - Scaling is applied to the input rather than the weights, but mathematically
      this is equivalent to StyleGAN's runtime weight scaling.

    Forward Pass
    ------------
    x : torch.Tensor
        Input tensor of shape (batch_size, in_features).

    Returns
    -------
    torch.Tensor
        Transformed tensor of shape (batch_size, out_features).
    """
    def __init__(self, in_features, out_features):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2/in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias

class PixelNorm(nn.Module):
    """
    Pixel-wise Feature Normalization Layer.

    Normalizes each feature vector across the channel dimension so that
    its mean squared magnitude is 1. This is commonly used in StyleGAN
    to stabilize training and remove dependence on the input scale.

    Formula:
        x_hat = x / sqrt(mean(x^2, dim=1, keepdim=True) + epsilon)

    Attributes
    ----------
    epsilon : float
        Small constant to avoid division by zero.
    """
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class MappingNetwork(nn.Module):
    """
    Mapping Network used in StyleGAN.

    Transforms an input latent vector z into an intermediate latent
    representation w. This allows disentangling of latent factors and
    provides more controllable style modulation in the generator.

    Parameters
    ----------
    input_dim : int
        Dimension of the input latent vector z.
    latent_dim : int
        Dimension of the output latent vector w.
    num_layers : int, optional (default=8)
        Number of fully connected layers in the mapping network.

    Attributes
    ----------
    mapping : nn.Sequential
        Sequential model consisting of PixelNorm, WSLinear layers, and ReLU activations.

    Forward Pass
    ------------
    x : torch.Tensor
        Input tensor of shape (batch_size, input_dim).

    Returns
    -------
    torch.Tensor
        Transformed latent vector w of shape (batch_size, latent_dim).
    """
    def __init__(self, input_dim, latent_dim, num_layers=8):
        """
        Mapping Network to transform input latent vectors/features to style vector w.
        Args:
            input_dim (int): Dimension of the input vector (e.g., features or noise vector z).
            latent_dim (int): Dimension of the output latent vector w.
            num_layers (int): Number of fully connected layers in the mapping network.
        """
        super(MappingNetwork, self).__init__()
        layers = [PixelNorm()]
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(WSLinear(in_dim, latent_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = latent_dim
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) Layer.

    This layer normalizes the input feature map per instance (channel-wise),
    and then applies a learned affine transformation using a latent style
    vector w. The style vector provides scale (gamma) and bias (beta)
    parameters that modulate the normalized features, allowing the generator
    to control image style (e.g., texture, color, patterns).

    Formula
    -------
    AdaIN(x, w) = scale(w) * InstanceNorm(x) + bias(w)

    Parameters
    ----------
    in_channels : int
        Number of channels in the input feature map.
    latent_dim : int
        Dimension of the style vector w.

    Attributes
    ----------
    instance_norm : nn.InstanceNorm2d
        Normalizes each channel of the input feature map.
    style_scale : WSLinear
        Maps latent vector w to per-channel scaling factors (gamma).
    style_bias : WSLinear
        Maps latent vector w to per-channel bias values (beta).

    Forward Pass
    ------------
    x : torch.Tensor
        Input feature map of shape (batch_size, in_channels, height, width).
    w : torch.Tensor
        Style vector of shape (batch_size, latent_dim).

    Returns
    -------
    torch.Tensor
        Style-modulated feature map of shape (batch_size, in_channels, height, width).
    """

    def __init__(self, in_channels, latent_dim):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(in_channels)
        self.style_scale = WSLinear(latent_dim, in_channels)
        self.style_bias = WSLinear(latent_dim, in_channels)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.instance_norm(x)  # normalize per channel
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)    # (B, C, 1, 1)

        return style_scale * x + style_bias

class WSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class ResBlock(nn.Module):
    def __init__(self, channels, latent_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)

        # Style-based normalization
        self.norm1 = AdaIN(channels, latent_dim)
        self.norm2 = AdaIN(channels, latent_dim)

    def forward(self, x, w=None):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out, w)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out, w)

        return out + residual

class ResNetGenerator(nn.Module):
    def __init__(self, z_dim, w_dim, img_channels=3, n_blocks=6, base_channels=64, num_layers=8, beta=0.995):
        super().__init__()
        self.map = MappingNetwork(z_dim, w_dim, num_layers)

        # w_avg trick
        self.register_buffer("w_avg", torch.zeros(1, w_dim))  # running average of w
        self.beta = beta

        # Initial 7x7 conv
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=False),
        )

        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=False),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=False),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(base_channels * 4, latent_dim=w_dim) for _ in range(n_blocks)
        ])

        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=False),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=False),
        )

        # Final conv
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, img_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x, z, use_w_avg=False):
        w = self.map(z)  # latent → style

        # Update running average (only in training mode)
        if self.training:
            batch_avg = w.mean(0, keepdim=True).detach()   # shape (1, w_dim)
            # in-place update to keep buffer registered
            self.w_avg.mul_(self.beta).add_(batch_avg * (1 - self.beta))

        # Optionally replace w with w_avg
        if use_w_avg:
            w = self.w_avg.expand(w.size(0), -1).contiguous()  # shape (B, w_dim)

        out = self.initial(x)
        out = self.down1(out)
        out = self.down2(out)

        for resblock in self.res_blocks:
            out = resblock(out, w)

        out = self.up1(out)
        out = self.up2(out)
        out = self.final(out)

        return out

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
class GANLoss(nn.Module):
    """Define GAN loss for LSGAN or BCEGAN"""
    def __init__(self, gan_mode='lsgan'):
        super().__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return torch.ones_like(prediction)
        else:
            return torch.zeros_like(prediction)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class VGGStyleLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(VGGStyleLoss, self).__init__()
        # Load pretrained VGG19 features
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Replace inplace ReLU with out-of-place ReLU
        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.ReLU):
                vgg[i] = nn.ReLU(inplace=False)

        self.vgg_layers = vgg.to(device).eval()

        # Freeze
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # Normalization
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).to(device)

        # Style feature layers
        self.style_layers = [2, 7, 12, 21, 30]

    def gram_matrix(self, feat):
        b, c, h, w = feat.size()
        feat = feat.view(b, c, h * w)                 # (B, C, H*W)
        G = torch.bmm(feat, feat.transpose(1, 2))     # (B, C, C)
        return G / (h * w)                            # normalize only by spatial size

    def forward(self, input, target):
        # [-1, 1] → [0, 1] → normalize
        input_norm = self.normalize((input + 1) / 2)
        target_norm = self.normalize((target + 1) / 2)

        style_loss = 0.0
        x_in, x_tg = input_norm, target_norm

        for i, layer in enumerate(self.vgg_layers):
            x_in = layer(x_in)
            x_tg = layer(x_tg)

            if i in self.style_layers:
                G_in = self.gram_matrix(x_in)
                G_tg = self.gram_matrix(x_tg)
                style_loss += F.mse_loss(G_in, G_tg)

        return style_loss

class UnetGenerator(nn.Module):
    """U-Net generator with AdaIN and style injection"""

    def __init__(self, input_nc, output_nc, num_downs, z_dim, w_dim, ngf=64, use_dropout=False, beta=0.995):
        """
        Args:
            input_nc (int)  -- input channels
            output_nc (int) -- output channels
            num_downs (int) -- number of downsamplings in U-Net
            z_dim (int)     -- latent dimension for style
            w_dim (int)     -- mapped style dimension
            ngf (int)       -- number of filters
        """

        super(UnetGenerator, self).__init__()

        self.map = MappingNetwork(z_dim, w_dim)   # latent → style

        # w_avg trick
        self.register_buffer("w_avg", torch.zeros(1, w_dim))  # running average of w
        self.beta = beta

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, 
                                             style_dim=w_dim,
                                             submodule=None,  
                                             innermost=True)  # add the innermost layer
        
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, 
                                                 style_dim=w_dim,
                                                 submodule=unet_block, 
                                                 use_dropout=use_dropout)
        
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, 
                                             style_dim=w_dim,
                                             submodule=unet_block)
        
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4,
                                             style_dim=w_dim,
                                             submodule=unet_block)
        
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2,
                                             style_dim=w_dim,
                                             submodule=unet_block)
        
        self.model = UnetSkipConnectionBlock(output_nc, ngf,
                                             style_dim=w_dim,
                                             input_nc=input_nc,
                                             submodule=unet_block,
                                             outermost=True)

    # Note: use_w_avg defaults to True here (False in ResNetGenerator.forward). The training and
    # test scripts rely on this default, so the U-Net generator always uses w_avg.
    def forward(self, x, z, use_w_avg=True):
        """Forward with latent injection"""
        w = self.map(z)  # latent → style

        # Update running average (only in training mode)
        if self.training:
            batch_avg = w.mean(0, keepdim=True).detach()   # shape (1, w_dim)
            # in-place update to keep buffer registered
            self.w_avg.mul_(self.beta).add_(batch_avg * (1 - self.beta))

        # Optionally replace w with w_avg
        if use_w_avg:
            w = self.w_avg.expand(w.size(0), -1).contiguous()  # shape (B, w_dim)

        return self.model(x, w)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with AdaIN skip connection"""

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 style_dim=None, submodule=None, 
                 outermost=False, innermost=False, 
                 use_dropout=False):

        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost
        self.submodule = submodule
        self.innermost = innermost
        self.style_dim = style_dim
        self.use_dropout = use_dropout

        if input_nc is None:
            input_nc = outer_nc
        
        # Build down path
        if innermost:
            down_layers = [
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1)
            ]
            if style_dim is not None:
                down_layers.append(AdaIN(inner_nc, style_dim))
        else:
            down_layers = [
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1)
            ]
            if style_dim is not None:
                down_layers.append(AdaIN(inner_nc, style_dim))

        # Build up path
        if outermost:
            up_layers = [
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
                nn.Tanh()
            ]
        elif innermost:
            up_layers = [
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            ]
            if style_dim is not None:
                up_layers.append(AdaIN(outer_nc, style_dim))
        else:
            up_layers = [
                nn.ReLU(inplace=False),
                nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            ]
            if style_dim is not None:
                up_layers.append(AdaIN(outer_nc, style_dim))

        self.down = nn.ModuleList(down_layers)
        self.up   = nn.ModuleList(up_layers)
        self.dropout = nn.Dropout(0.5) if use_dropout else None

    def _apply_layers(self, layers, x, w):
        out = x
        for layer in layers:
            if isinstance(layer, AdaIN):
                out = layer(out, w)   # AdaIN(features, w)
            else:
                out = layer(out)
        return out

    def forward(self, x, w):
        # down
        out = self._apply_layers(self.down, x, w)

        # recurse
        if self.submodule is not None:
            out = self.submodule(out, w)

        # up
        out = self._apply_layers(self.up, out, w)

        if self.outermost:
            return out

        if self.dropout is not None:
            out = self.dropout(out)

        # skip connection: concat input x with up-output
        return torch.cat([x, out], dim=1)