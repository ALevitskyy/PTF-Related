import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


def get_channels(architecture):
    if architecture in ["resnet18", "resnet34"]:
        return [512, 256, 128, 64]
    elif architecture in ["resnet50", "resnet101", "resnet152", "se_resnext50_32x4d"]:
        return [2048, 1024, 512, 256]
    else:
        raise Exception("architecture is not supported as backbone")


class ConvRelu(nn.Module):
    """3x3 convolution followed by ReLU activation building block.
    """

    def __init__(self, num_in, num_out):
        """Creates a `ConvReLU` building block.
        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """
        The networks forward pass for which
            autograd synthesizes the backwards pass.
        Args:
          x: the input tensor
        Returns:
          The networks output tensor.
        """

        return F.relu(self.block(x), inplace=True)


class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
        )

        self.down2_1 = nn.Sequential(
            nn.Conv2d(
                input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False
            ),
            nn.BatchNorm2d(input_dim),
            nn.ELU(True),
        )
        self.down2_2 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ELU(True),
        )

        self.down3_1 = nn.Sequential(
            nn.Conv2d(
                input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(input_dim),
            nn.ELU(True),
        )
        self.down3_2 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ELU(True),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ELU(True),
        )

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.upsample(
            x_glob, scale_factor=16, mode="bilinear", align_corners=True
        )  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.upsample(
            d3, scale_factor=2, mode="bilinear", align_corners=True
        )  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.upsample(
            d2, scale_factor=2, mode="bilinear", align_corners=True
        )  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x


def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(
        nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size=3,
            dilation=rate,
            padding=rate,
            bias=False,
        ),
        nn.BatchNorm2d(output_dim),
        nn.ELU(True),
    )


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            input_dim, input_dim // reduction, kernel_size=1, stride=1
        )
        self.conv2 = nn.Conv2d(
            input_dim // reduction, input_dim, kernel_size=1, stride=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class DecoderBlock(nn.Module):
    """Decoder building block upsampling resolution by a factor of two.
    """

    def __init__(self, num_in, num_out):
        """Creates a `DecoderBlock` building block.
        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = ConvRelu(num_in, num_out)
        self.s_att = SpatialAttention2d(num_out)
        self.c_att = GAB(num_out, 16)
        self.bn = nn.BatchNorm2d(num_out)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        """
        The networks forward pass for which
            autograd synthesizes the backwards pass.
        Args:
          x: the input tensor
        Returns:
          The networks output tensor.
        """
        first = self.block(F.interpolate(x, scale_factor=2, mode="nearest"))
        cat_p = self.relu(self.bn(first))
        s = self.s_att(cat_p)
        c = self.c_att(cat_p)
        return s + c


class ResNetUnet(nn.Module):
    """
    U-Net inspired encoder-decoder architecture for semantic segmentation,
        with a ResNet encoder as proposed by Alexander Buslaev.
    Also known as AlbuNet
    See:
    - https://arxiv.org/abs/1505.04597 -
        U-Net: Convolutional Networks for Biomedical Image Segmentation
    - https://arxiv.org/abs/1411.4038  -
        Fully Convolutional Networks for Semantic Segmentation
    - https://arxiv.org/abs/1512.03385 -
        Deep Residual Learning for Image Recognition
    - https://arxiv.org/abs/1801.05746 -
        TernausNet: U-Net with VGG11
        Encoder Pre-Trained on ImageNet for Image Segmentation
    - https://arxiv.org/abs/1806.00844 -
        TernausNetV2: Fully Convolutional Network for Instance Segmentation
    based on https://github.com/mapbox/robosat/blob/master/robosat/unet.py
    """

    def __init__(
        self, num_classes=1, num_filters=32, backbone="resnet34", pretrained=True
    ):
        """Creates an `UNet` instance for semantic segmentation.
        Args:
          num_classes: number of classes to predict.
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()

        # Todo: make input channels configurable,
        # not hard-coded to three channels for RGB

        self.resnet = pretrainedmodels.__dict__[backbone](
            num_classes=1000, pretrained="imagenet"
        )
        encoder_channels = get_channels(backbone)

        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.center = DecoderBlock(encoder_channels[0], num_filters * 8)

        self.dec0 = DecoderBlock(
            num_in=encoder_channels[0] + num_filters * 8, num_out=num_filters * 8
        )
        self.dec1 = DecoderBlock(
            num_in=encoder_channels[1] + num_filters * 8, num_out=num_filters * 8
        )
        self.dec2 = DecoderBlock(
            num_in=encoder_channels[2] + num_filters * 8, num_out=num_filters * 2
        )
        self.dec3 = DecoderBlock(
            num_in=encoder_channels[3] + num_filters * 2, num_out=num_filters * 2 * 2
        )
        self.dec4 = DecoderBlock(num_filters * 2 * 2, num_filters)
        self.dec5 = ConvRelu(num_filters, num_filters)

        self.logit = nn.Sequential(
            nn.Conv2d(768, 64, kernel_size=3, padding=1),
            nn.ELU(True),
            nn.Conv2d(64, 1, kernel_size=1, bias=False),
        )

    def forward(self, x):
        """
        The networks forward pass for which
            autograd synthesizes the backwards pass.
        Args:
          x: the input tensor
        Returns:
          The networks output tensor.
        """
        size = x.size()
        assert (
            size[-1] % 64 == 0 and size[-2] % 64 == 0
        ), "image resolution has to be divisible by 64 for resnet"

        # enc0 = self.resnet.conv1(x)
        # enc0 = self.resnet.bn1(enc0)
        # enc0 = self.resnet.relu(enc0)
        # enc0 = self.resnet.maxpool(enc0)
        enc0 = self.resnet.layer0(x)
        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        center = self.center(F.max_pool2d(enc4, kernel_size=2, stride=2))

        dec0 = self.dec0(torch.cat([enc4, center], dim=1))
        dec1 = self.dec1(torch.cat([enc3, dec0], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc1, dec2], dim=1))
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)
        f = torch.cat(
            (
                dec5,
                dec4,
                F.upsample(dec3, scale_factor=2, mode="bilinear", align_corners=True),
                F.upsample(dec2, scale_factor=4, mode="bilinear", align_corners=True),
                F.upsample(dec1, scale_factor=8, mode="bilinear", align_corners=True),
                F.upsample(dec0, scale_factor=16, mode="bilinear", align_corners=True),
            ),
            1,
        )  # 320, 256, 256
        logit = self.logit(f)

        return logit
