import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ConvBlock(nn.Module):
    """(Conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNetPP(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, deep_supervision=False, base_filters=32, fusion=False):
        super(UNetPP, self).__init__()
        self.deep_supervision = deep_supervision
        self.fusion = fusion

        nb_filter = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, base_filters * 16]

        # Encoder
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        self.pool = nn.MaxPool2d(2)

        # Decoder (nested)
        self.up1_0 = Up(nb_filter[1], nb_filter[0])
        self.up2_0 = Up(nb_filter[2], nb_filter[1])
        self.up3_0 = Up(nb_filter[3], nb_filter[2])
        self.up4_0 = Up(nb_filter[4], nb_filter[3])

        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[0], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[1], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[2], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[3], nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[0], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[1], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[2], nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[0], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[1], nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[0], nb_filter[0])

        # Deep supervision heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Decoder path (nested connections)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_0(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_0(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_0(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_0(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_0(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3)], 1))

        if self.fusion:
            return x0_4
        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


# # ------------------- Basic Conv Block -------------------
# class ConvBNGelu(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.GELU(),
#             nn.Dropout2d(0.1)
#         )
#
#     def forward(self, x):
#         return self.block(x)
#
#
# # ------------------- Encoder -------------------
# class Encoder(nn.Module):
#     def __init__(self, in_channels=3, base_band=64, num_layers=4):
#         super().__init__()
#         self.bands = [base_band * (2 ** i) for i in range(num_layers)]  # e.g. [64,128,256,512]
#
#         self.layer1 = nn.Sequential(
#             ConvBNGelu(in_channels, self.bands[0]),
#             ConvBNGelu(self.bands[0], self.bands[0])
#         )
#         self.pool1 = nn.MaxPool2d(2, 2)
#
#         self.layer2 = nn.Sequential(
#             ConvBNGelu(self.bands[0], self.bands[1]),
#             ConvBNGelu(self.bands[1], self.bands[1])
#         )
#         self.pool2 = nn.MaxPool2d(2, 2)
#
#         self.layer3 = nn.Sequential(
#             ConvBNGelu(self.bands[1], self.bands[2]),
#             ConvBNGelu(self.bands[2], self.bands[2])
#         )
#         self.pool3 = nn.MaxPool2d(2, 2)
#
#         # Last layer with dilation for high-level features
#         self.layer4 = nn.Sequential(
#             ConvBNGelu(self.bands[2], self.bands[3], dilation=2, padding=2),
#             ConvBNGelu(self.bands[3], self.bands[3], dilation=2, padding=2)
#         )
#
#     def forward(self, x):
#         x1 = self.layer1(x)  # low-level feature
#         x2 = self.layer2(self.pool1(x1))
#         x3 = self.layer3(self.pool2(x2))
#         x4 = self.layer4(self.pool3(x3))  # high-level feature
#         return x1, x4
#
#
# # ------------------- ASPP Module -------------------
# class ASPP(nn.Module):
#     def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
#         super().__init__()
#         self.aspp_blocks = nn.ModuleList([
#             ConvBNGelu(in_ch, out_ch, kernel_size=1, padding=0, dilation=rates[0]),
#             ConvBNGelu(in_ch, out_ch, kernel_size=3, padding=rates[1], dilation=rates[1]),
#             ConvBNGelu(in_ch, out_ch, kernel_size=3, padding=rates[2], dilation=rates[2]),
#             ConvBNGelu(in_ch, out_ch, kernel_size=3, padding=rates[3], dilation=rates[3])
#         ])
#         self.global_pool = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             ConvBNGelu(in_ch, out_ch, kernel_size=1, padding=0)
#         )
#         self.project = nn.Sequential(
#             nn.Conv2d(out_ch * 5, out_ch, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.GELU(),
#             nn.Dropout(0.5)
#         )
#
#     def forward(self, x):
#         size = x.shape[2:]
#         res = [conv(x) for conv in self.aspp_blocks]
#         gp = self.global_pool(x)
#         gp = F.interpolate(gp, size=size, mode='bilinear', align_corners=False)
#         res.append(gp)
#         x = torch.cat(res, dim=1)
#         return self.project(x)
#
#
# # ------------------- Decoder -------------------
# class Decoder(nn.Module):
#     def __init__(self, low_ch, high_ch, out_ch, reduce_low_ch=None, mid_ch=None):
#         """
#         low_ch: channels of low-level encoder feature (encoder.bands[0])
#         high_ch: channels of high-level ASPP output
#         out_ch: number of classes (final channels)
#
#         Optional:
#         reduce_low_ch: how many channels to reduce low-level feature to (default: low_ch // 2, min 8)
#         mid_ch: mid/fusion channels after concatenation (default: max(64, high_ch))
#         """
#         super().__init__()
#
#         # sensible defaults that scale with base_band:
#         if reduce_low_ch is None:
#             reduce_low_ch = max(8, low_ch // 2)  # replaces hard-coded 48
#         if mid_ch is None:
#             mid_ch = max(64, high_ch)  # replaces hard-coded 256
#
#         self.reduce_low = ConvBNGelu(low_ch, reduce_low_ch, kernel_size=1, padding=0)
#         self.fuse = nn.Sequential(
#             ConvBNGelu(high_ch + reduce_low_ch, mid_ch),
#             ConvBNGelu(mid_ch, mid_ch)
#         )
#         self.final = nn.Conv2d(mid_ch, out_ch, kernel_size=1)
#
#     def forward(self, low_feat, high_feat, return_features):
#         low_feat = self.reduce_low(low_feat)
#         high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
#         x = torch.cat([low_feat, high_feat], dim=1)
#         fused = self.fuse(x)
#         return fused if return_features else self.final(fused)
#
#
# # ------------------- DeepLabV3+ Model -------------------
# class DeepLabV3Plus(nn.Module):
#     def __init__(self, in_channels=3, num_classes=4, base_band=32, return_features=False):
#         """
#         base_band: root channel width (e.g. 32). Changing this will propagate:
#             - encoder.bands[0] becomes base_band
#             - ASPP output out_ch is derived as base_band * 4 (keeps your original ratio)
#             - Decoder's reduce_low_ch and mid_ch are derived from low_ch/high_ch automatically
#         """
#         super().__init__()
#         self.return_features = return_features  # Flag for fusion
#         self.encoder = Encoder(in_channels, base_band=base_band)
#
#         # Keep your original choice: ASPP out channels = base_band * 4 (but derived from base_band)
#         asp_out = base_band * 4
#         self.aspp = ASPP(in_ch=self.encoder.bands[-1], out_ch=asp_out)
#
#         # Decoder will auto-select sensible reduce_low and mid channels based on low_ch and asp_out
#         self.decoder = Decoder(low_ch=self.encoder.bands[0], high_ch=asp_out, out_ch=num_classes)
#
#     def forward(self, x):
#         low_feat, high_feat = self.encoder(x)
#         high_feat = self.aspp(high_feat)
#         out = self.decoder(low_feat, high_feat, self.return_features)
#
#         if self.return_features:
#             return out  # For fusion, don't interpolate
#         else:
#             # Normal standalone DeepLab output
#             out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
#             return out
#
#
# class GlacierNetBeta(nn.Module):
#     """
#     Fusion model: UNet++ and DeepLabV3+ in parallel.
#     Simplified version: replaces Channel Attention Fusion with plain concatenation-based fusion.
#     """
#
#     def __init__(self, in_channels=5, out_channels=4, base_band=32):
#         super(GlacierNetBeta, self).__init__()
#
#         # UNet++ with fusion=True to get intermediate feature map
#         self.unetpp = UNetPP(in_channels=in_channels, out_channels=out_channels, fusion=True, base_filters=base_band)
#
#         # DeepLabV3+ (forward returns feature map, not resized output)
#         self.deeplabv3 = DeepLabV3Plus(in_channels=in_channels, num_classes=out_channels, base_band=base_band,
#                                        return_features=True)
#
#         # Derived channel sizes
#         C1 = base_band  # UNet++ feature channels (conv0_4 output)
#         C2 = base_band * 4  # DeepLab ASPP output channels
#         fusion_in = C1 + C2  # total after concatenation
#
#         mid = max(base_band * 2, out_channels * 2)
#         self.fuse_conv = nn.Sequential(
#             nn.Conv2d(fusion_in, mid, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(mid),
#             nn.GELU(),
#             nn.Dropout2d(0.15),
#             nn.Conv2d(mid, out_channels, kernel_size=1)
#         )
#
#     def forward(self, x):
#         # Parallel feature extraction
#         feat_unetpp = self.unetpp(x)
#         feat_deeplab = self.deeplabv3(x)
#
#         # Resize DeepLab output to match UNet++ size
#         if feat_unetpp.shape[2:] != feat_deeplab.shape[2:]:
#             feat_deeplab = F.interpolate(feat_deeplab, size=feat_unetpp.shape[2:], mode='bilinear', align_corners=False)
#
#         # Concatenate along channel dimension instead of attention fusion
#         fused = torch.cat([feat_unetpp, feat_deeplab], dim=1)
#
#         # Fuse and reduce to final segmentation
#         out = self.fuse_conv(fused)
#         return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNetPP(5, 4, base_filters=16)
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        x = torch.randn(16, 5, 128, 128)
        x = x.to(device)
        out = model(x)
        print(out.shape)

        summary(model, input_size=(5, 128, 128))
    # torch.save(model.state_dict(), "model.pth")
