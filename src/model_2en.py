import torch
import torch.nn as nn

# -----------------------------
# Base building blocks
# -----------------------------

class ConvBlock(nn.Module):
    """(Conv => BN => GELU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
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
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


# -----------------------------
# Main UNet++ with Multimodal Encoder
# -----------------------------

class UNetPP_Multimodal(nn.Module):
    def __init__(self, in_channels=5, out_channels=4, base_filters=32, deep_supervision=False):
        super().__init__()
        assert in_channels == 5, "Expected 5-channel input (3 RGB + 2 IR)"
        self.deep_supervision = deep_supervision

        nb_filter = [base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16]

        # -----------------------------
        # RGB Encoder (3 channels)
        # -----------------------------
        self.rgb_conv0_0 = ConvBlock(3, nb_filter[0])
        self.rgb_conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.rgb_conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.rgb_conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.rgb_conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        # -----------------------------
        # IR Encoder (2 channels)
        # -----------------------------
        self.ir_conv0_0 = ConvBlock(2, nb_filter[0])
        self.ir_conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.ir_conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.ir_conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.ir_conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        self.pool = nn.MaxPool2d(2)

        # -----------------------------
        # Fusion: concat RGB + IR features
        # -----------------------------
        def fuse_block(in_ch):
            return ConvBlock(in_ch * 2, in_ch)

        self.fuse0 = fuse_block(nb_filter[0])
        self.fuse1 = fuse_block(nb_filter[1])
        self.fuse2 = fuse_block(nb_filter[2])
        self.fuse3 = fuse_block(nb_filter[3])
        self.fuse4 = fuse_block(nb_filter[4])

        # -----------------------------
        # Decoder (standard UNet++)
        # -----------------------------
        self.up1_0 = Up(nb_filter[1], nb_filter[0])
        self.up2_0 = Up(nb_filter[2], nb_filter[1])
        self.up3_0 = Up(nb_filter[3], nb_filter[2])
        self.up4_0 = Up(nb_filter[4], nb_filter[3])

        self.conv0_1 = ConvBlock(nb_filter[0]*2, nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1]*2, nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2]*2, nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3]*2, nb_filter[3])

        self.conv0_2 = ConvBlock(nb_filter[0]*3, nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*3, nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*3, nb_filter[2])

        self.conv0_3 = ConvBlock(nb_filter[0]*4, nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*4, nb_filter[1])

        self.conv0_4 = ConvBlock(nb_filter[0]*5, nb_filter[0])

        # -----------------------------
        # Deep supervision heads or single head
        # -----------------------------
        if deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, 1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, 1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, 1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, 1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, 1)

    def forward(self, x):
        # Split input
        x_rgb = x[:, :3, :, :][:, [2, 1, 0], :, :]
        x_ir = x[:, 3:, :, :]

        # RGB encoder
        x0_0_r = self.rgb_conv0_0(x_rgb)
        x1_0_r = self.rgb_conv1_0(self.pool(x0_0_r))
        x2_0_r = self.rgb_conv2_0(self.pool(x1_0_r))
        x3_0_r = self.rgb_conv3_0(self.pool(x2_0_r))
        x4_0_r = self.rgb_conv4_0(self.pool(x3_0_r))

        # IR encoder
        x0_0_i = self.ir_conv0_0(x_ir)
        x1_0_i = self.ir_conv1_0(self.pool(x0_0_i))
        x2_0_i = self.ir_conv2_0(self.pool(x1_0_i))
        x3_0_i = self.ir_conv3_0(self.pool(x2_0_i))
        x4_0_i = self.ir_conv4_0(self.pool(x3_0_i))

        # Fuse at each encoder level via concat
        x0_0 = self.fuse0(torch.cat([x0_0_r, x0_0_i], dim=1))
        x1_0 = self.fuse1(torch.cat([x1_0_r, x1_0_i], dim=1))
        x2_0 = self.fuse2(torch.cat([x2_0_r, x2_0_i], dim=1))
        x3_0 = self.fuse3(torch.cat([x3_0_r, x3_0_i], dim=1))
        x4_0 = self.fuse4(torch.cat([x4_0_r, x4_0_i], dim=1))

        # -----------------------------
        # UNet++ decoding path
        # -----------------------------
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

        # -----------------------------
        # Output
        # -----------------------------
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)


# -----------------------------
# Sanity check
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 5, 128, 128)  # 3 RGB + 2 IR
    x = x.to(device)
    model = UNetPP_Multimodal(in_channels=5, out_channels=4, base_filters=32)
    model.to(device)
    model.eval()
    y = model(x)
    print("Output shape:", y.shape)
    from torchinfo import summary

    summary(model, input_size=(1, 5, 128, 128), col_names=["input_size", "output_size", "num_params"])
