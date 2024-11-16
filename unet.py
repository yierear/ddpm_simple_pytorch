from torch import nn
import torch
import math

from dataloader import show_tensor_image


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)  # 特征维度对齐
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)  # * 2因为用了skip connection
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)  # 恢复图片尺寸，放大2倍
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  # 缩小两倍
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)  # 尺寸不变
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, time_emb):
        x = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(time_emb))
        time_emb = time_emb[(...,) + (None,) * 2]  # 维度对齐
        x = x + time_emb
        x = self.bnorm2(self.relu(self.conv2(x)))
        return self.transform(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        # 计算频率因子
        embeddings = math.log(10000) / (half_dim - 1)  # 计算缩放系数
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList(
            [Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)]
        )
        # Upsample
        self.ups = nn.ModuleList(
            [Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)]
        )

        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        x = self.conv0(x)
        # Unet
        skip_inputs = []
        for down in self.downs:
            x = down(x, t)
            skip_inputs.append(x)
        for up in self.ups:
            skip_x = skip_inputs.pop()
            # 跳跃连接
            x = torch.cat((x, skip_x), dim=1)
            x = up(x, t)
        return self.output(x)


if __name__ == "__main__":
    model = SimpleUnet()
    print("参数量: ", sum(p.numel() for p in model.parameters()))
    print(model)
    img_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.randn((1, 3, img_size, img_size), device=device)
    t = torch.randint(0, 10, (1,), device=device)
    output = model(img, t)

    # show_tensor_image(output.detach().cpu(), is_show=True)