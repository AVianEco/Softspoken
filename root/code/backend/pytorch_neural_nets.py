import torch
import torchaudio
import torch.nn as nn

from root.code.backend import settings

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, dropout_p = 0.1):
        super(ResBlock, self).__init__()
        padding = kernel_size // 2

        # 1x1 conv to match channels for the residual
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Main path
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout2d(p = dropout_p)

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        
        out = out + identity

        out = self.relu(out)
        out = self.dropout(out)
        return out

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, dropout_p = 0.1):
        super(ResBlock1D, self).__init__()
        padding = kernel_size // 2

        # Residual path: always project using 1x1 convolution
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        # Main path
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout1d(p = dropout_p)

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        
        out = out + identity

        out = self.relu(out)
        out = self.dropout(out)
        return out

class SpecUNet_2D(nn.Module):
    def sqrt_log10_nonzero(self, magnitudes):
        return torch.sqrt(torch.log10(magnitudes + 1))

    def __init__(self):
        super(SpecUNet_2D, self).__init__()

        filters = 32
        self.n_mels = 128  # Number of mel bins

        self.input_shape = (66150)
        self.output_shape = (2, 128, 256)
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=settings.vad_resample,
            n_fft=settings.n_fft *4,
            win_length=settings.win_length,
            hop_length=settings.hop_length,
            n_mels=self.n_mels,
            f_max=8000
        )

        # Encoder
        self.conv1_1 = ResBlock(1, filters * 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = ResBlock(filters * 1, filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = ResBlock(filters * 2, filters * 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = ResBlock(filters * 3, filters * 4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv_bottleneck = ResBlock(filters * 4, filters * 4)
        self.encoder_out = ResBlock(filters * 4, filters * 4)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

        # Decoder
        self.conv6 = ResBlock(filters * 8, filters * 3)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv7 = ResBlock(filters * 6, filters * 2)
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = ResBlock(filters * 4, filters * 1)
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv9_1 = ResBlock(filters * 2, filters * 1)

        # env / speech separation
        self.spec_output_conv = nn.Sequential(
            ResBlock(filters, filters),
            nn.Conv2d(filters, 2, kernel_size=1)
        )
        self.spec_output_activation = nn.ReLU()

        # Flattening and 1D convolutions
        self.conv_flatten = nn.Conv2d(filters * 1, 4, kernel_size=(self.n_mels, 1))
        self.relu_flatten = nn.ReLU()

        # Output layer for mask
        self.mask_output_conv =  nn.Sequential(
            ResBlock1D(4, 4),
            nn.Conv1d(4, 1, kernel_size=1)
        )
     
    def forward(self, x):
        # Compute the mel spectrogram: (batch_size, n_mels, time_steps)
        mel_spec = self.mel_spectrogram(x)

        # Apply sqrt_log10_nonzero scalingw
        mel_spec = self.sqrt_log10_nonzero(mel_spec)
        
        # Trim the time dimension to 256 frames
        mel_spec = mel_spec[:, :, :256]  # Shape: (batch_size, n_mels, 256)

        # Add a singleton channel dimension
        mel_spec = mel_spec.unsqueeze(1)  # Shape: (batch_size, 1, n_mels, 256)

        # Encoder
        conv1 = self.conv1_1(mel_spec)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2_1(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3_1(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4_1(pool3)
        pool4 = self.pool4(conv4)

        # Bottleneck
        conv_bottleneck = self.conv_bottleneck(pool4)
        encoder_out = self.encoder_out(conv_bottleneck)
        encoder_up = self.up1(encoder_out)

        # Decoder
        merge1 = torch.cat([conv4, encoder_up], dim=1)
        conv6 = self.conv6(merge1)
        up2 = self.up2(conv6)
        merge2 = torch.cat([conv3, up2], dim=1)
        conv7 = self.conv7(merge2)
        up3 = self.up3(conv7)
        merge3 = torch.cat([conv2, up3], dim=1)
        conv8 = self.conv8(merge3)
        up4 = self.up4(conv8)
        merge4 = torch.cat([conv1, up4], dim=1)
        conv9 = self.conv9_1(merge4)

        # Output for spec_output
        spec_output = self.spec_output_conv(conv9) 
        spec_output = self.spec_output_activation(spec_output)

        # Flatten along frequency dimension
        x_flat = self.conv_flatten(conv9)
        x_flat = self.relu_flatten(x_flat)

        # Remove the frequency dimension
        x_flat = x_flat.squeeze(2)  

        # Output for mask_output
        mask_output = self.mask_output_conv(x_flat)

        return spec_output, mask_output

