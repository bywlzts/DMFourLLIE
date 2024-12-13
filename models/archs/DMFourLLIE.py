import functools
import torch
import torch.nn as nn
import models.archs.arch_util as arch_util
from models.archs.luminance_map import LuminaceMap
from models.archs.ffc import FFCResnetBlock
import torch.nn.functional as F

class DMFourLLIE(nn.Module):
    def __init__(self, y_nf=16, f_nf=16, s_nf=32):
        """
        DMFourLLIE
        Args:
            y_nf (int): Number of channels for luminance map processing.
            f_nf (int): Number of channels for Fourier stage processing.
            s_nf (int): Number of channels for multi-stage processing.
        """
        super(DMFourLLIE, self).__init__()
        self.y_nf = y_nf
        self.f_nf = f_nf
        self.s_nf = s_nf

        # Luminance map processing
        self.lattnet = LuminaceMap(depth=[1, 1, 1, 1], base_channel=self.y_nf)
        # Fourier stage processing
        self.fourstage = FirstProcessModel(nf=self.f_nf)
        # Multi-stage processing
        self.ffcmultistage = SecondProcessModel(nf=self.s_nf, num_blocks=6, input_channels=3)
        # Initial convolution layers for Fourier features
        self.conv_first_fr = nn.Conv2d(3, self.f_nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_first_map = nn.Conv2d(1, self.f_nf, kernel_size=1, stride=1, padding=0, bias=True)
        # Activation function
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def pad_to_multiple(self, x, multiple=8):
        _, _, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x

    def get_amplitude(self, x, fr, y_map):
        # Fourier transform of infrared image

        fr_fft = torch.fft.rfft2(fr, norm='backward')
        pha_fr = torch.angle(fr_fft)  # Avoid NaN
        pha_fr = self.conv_first_fr(pha_fr)

        # Fourier transform of luminance map
        mp_fft = torch.fft.rfft2(y_map, norm='backward')
        amp_mp = torch.abs(mp_fft)
        amp_mp = self.conv_first_map(amp_mp)

        # Amplitude enhancement
        x_amplitude, y_map_out = self.fourstage(x, pha_fr, amp_mp)

        # Pad to make dimensions divisible by 8
        x_amplitude = self.pad_to_multiple(x_amplitude, multiple=8)
        x = self.pad_to_multiple(x, multiple=8)

        return x_amplitude, x, pha_fr

    def forward(self, x, fr, Y):

        # Luminance map processing
        y_map = self.lattnet(Y)
        # Amplitude enhancement
        x_four, x, pha_fr = self.get_amplitude(x, fr, y_map)

        # Multi-stage processing
        out_noise = self.ffcmultistage(x, x_four)

        return out_noise, x_four, y_map

class FirstProcessModel(nn.Module):
    def __init__(self, nf):
        """
        First Stage --- Fourier Reconstruction Stage.
        """
        super(FirstProcessModel, self).__init__()

        # Initial feature extraction
        self.initial_conv = nn.Conv2d(3, nf, kernel_size=1, stride=1, padding=0)

        # FFT-based feature extraction blocks
        self.fft_blocks = nn.ModuleList([FFT_Process(nf) for _ in range(6)])

        # Downsampling layers for feature fusion
        self.concat_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0),
                arch_util.SEBlock(nf)
            ) for _ in range(3)
        ])
        # Final output layer
        self.output_conv = nn.Conv2d(nf, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x, fr, y_map):
        # Initial feature extraction

        x0 = self.initial_conv(x)
        # FFT-based feature extraction with skip connections
        x, fr, y_map = self.fft_blocks[0](x0,fr,y_map)
        x1, fr, y_map = self.fft_blocks[1](x,fr,y_map)
        x2, fr, y_map = self.fft_blocks[2](x1,fr,y_map)

        # Downsample and fuse features
        x3_input = torch.cat((x2, x1), dim=1)
        x3_input = self.concat_layers[0](x3_input)
        x3, fr, y_map = self.fft_blocks[3](x3_input,fr,y_map)

        x4_input = torch.cat((x3, x), dim=1)
        x4_input = self.concat_layers[1](x4_input)
        x4, fr, y_map = self.fft_blocks[4](x4_input,fr,y_map)

        x5_input = torch.cat((x4, x0), dim=1)
        x5_input = self.concat_layers[1](x5_input)
        x5, fr, y_map = self.fft_blocks[5](x5_input,fr,y_map)

        # Final output
        xout = self.output_conv(x5)
        return xout, y_map

class SecondProcessModel(nn.Module):
    def __init__(self, nf=64, num_blocks=6, input_channels=3):
        """
        Second Stage --- Spatial and Texture Reconstruction Stage.
        """
        super(SecondProcessModel, self).__init__()

        # Initial convolution layers (downsampling)
        self.conv1 = nn.Conv2d(input_channels * 2, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Dual-branch blocks
        self.fft_blocks = nn.ModuleList([FFCResnetBlock(nf) for _ in range(num_blocks)])
        self.multi_blocks = nn.ModuleList([MultiConvBlock(nf) for _ in range(num_blocks)])
        self.fusion_blocks = nn.ModuleList([arch_util.ChannelAttentionFusion(nf) for _ in range(num_blocks)])

        # Reconstruction trunk
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, 1)

        # Upsample convolution layers
        self.upconv1 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.upconv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)


    def forward(self, x, x_four):
        # Downsampling with intermediate feature extraction
        x1 = self.lrelu(self.conv1(torch.cat((x, x_four), dim=1)))  # First downsampling
        x2 = self.lrelu(self.conv2(x1))  # Second downsampling
        x3 = self.lrelu(self.conv3(x2))  # Third downsampling
        # Dual-branch processing
        fft_features = x3
        multi_features = x3

        for fft_block, multi_block, fusion_block in zip(self.fft_blocks, self.multi_blocks, self.fusion_blocks):
            fft_features = fft_block(fft_features)
            multi_features = multi_block(multi_features)
            # Fuse features using Channel Attention Fusion
            fused_features = fusion_block(fft_features, multi_features)
        # Reconstruction and upsampling
        out_noise = self.recon_trunk(fused_features)
        out_noise = self._upsample(out_noise, x3, self.upconv1)
        out_noise = self._upsample(out_noise, x2, self.upconv2)
        out_noise = self.upconv3(torch.cat((out_noise, x1), dim=1))
        out_noise = self.upconv_last(out_noise)
        # Residual connection
        out_noise = out_noise + x
        # Ensure output size matches input size
        B, C, H, W = x.size()
        out_noise = out_noise[:, :, :H, :W]

        return out_noise

    def _upsample(self, x, skip, upconv):
        x = self.lrelu(self.pixel_shuffle(upconv(torch.cat((x, skip), dim=1))))
        return x

class MultiConvBlock(nn.Module):
    def __init__(self, dim, num_heads=4, expand_ratio=2):
        super(MultiConvBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Channel reduction layer
        self.conv_reduction = nn.Conv2d(dim, dim // 4, kernel_size=1, stride=1, bias=True)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

        # Multi-scale convolution layers
        self.local_convs = nn.ModuleList([
            nn.Conv2d(
                dim // 4, dim // 4,
                kernel_size=(3 + i * 2),
                padding=(1 + i),
                stride=1,
                groups=dim // 4  # Grouped convolution
            ) for i in range(num_heads)
        ])

        # Feature fusion layer
        self.conv_fusion = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=True)
        self.se_block = arch_util.SEBlock(dim)

    def forward(self, x):
        # Channel reduction
        x_reduced = self.leakyrelu(self.conv_reduction(x))

        # Multi-scale feature extraction
        multi_scale_features = []
        for conv in self.local_convs:
            x_scale = self.leakyrelu(conv(x_reduced))  # Apply multi-scale convolution
            x_scale = x_scale * torch.sigmoid(x_reduced)  # Element-wise modulation
            multi_scale_features.append(x_scale)

        # Concatenate multi-scale features
        x_concat = torch.cat(multi_scale_features, dim=1)

        # Feature fusion and residual connection
        x_fused = self.conv_fusion(x_concat)
        x_fused = self.se_block(x_fused)
        return x + x_fused  # Residual connection

class FFT_Process(nn.Module):
    def __init__(self, nf):
        super(FFT_Process, self).__init__()
        # Preprocessing for frequency domain
        self.nf = nf
        self.freq_preprocess = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        self.feature_fusion = nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0)
        self.process_amp = self._make_process_block(nf)
        self.process_pha = self._make_process_block(nf)
        self.process_fr = self._make_process_block(nf)
        self.process_map = self._make_process_block(nf)

    def _make_process_block(self, nf):
        return nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0)
        )

    def _make_fr_map_block(self, x, nf):
        B, C, H, W = x.shape
        if C == nf:
            return x
        elif C < nf:
            repeat_factor = nf // C + (1 if nf % C != 0 else 0)
            x = x.repeat(1, repeat_factor, 1, 1)[:, :nf, :, :]
        else:
            x = torch.nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
            x = x[:, :nf, :, :]
        return x


    def multiply_and_softmax(self, vis, fra):
        # Normalize features to avoid numerical instability
        vis = F.normalize(vis, dim=1)
        fra = F.normalize(fra, dim=1)

        # Flatten and multiply
        features1_flattened = vis.view(vis.size(0), vis.size(1), -1)
        features2_flattened = fra.view(fra.size(0), fra.size(1), -1)
        multiplied = torch.mul(features1_flattened, features2_flattened)

        # Apply softmax
        multiplied_softmax = torch.softmax(multiplied, dim=2)
        multiplied_softmax = multiplied_softmax.view(vis.size(0), vis.size(1), vis.size(2), vis.size(3))

        # Residual connection
        vis_map = vis * multiplied_softmax + vis
        return vis_map

    def forward(self, x, fr, y_map):
        _, _, H, W = x.shape
        # Frequency domain processing
        x_freq = torch.fft.rfft2(self.freq_preprocess(x), norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process_amp(mag)
        pha = self.process_pha(pha)

        # Process infrared features and Cross-modality interaction
        fr = self.process_fr(fr)
        pha = self.multiply_and_softmax(pha, fr)

        # Process brightness attention map
        y_map = torch.sigmoid(self.process_map(y_map))
        mag = mag * y_map + mag

        # Reconstruct frequency domain features
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        x_out_ff = x_out + x
        return x_out_ff, fr, y_map
