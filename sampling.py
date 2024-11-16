import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from forward_noising import (
    T,
    betas,
    alphas,
    alphas_cumprod,
    get_index_from_list,
    sqrt_one_minus_alphas_cumprod,
)
import matplotlib.pyplot as plt
from dataloader import show_tensor_image
from unet import SimpleUnet

# 定义常量
# alphas_cumprod_prev 是 alphas_cumprod 的前一时间步的值，并在最前面补 1（表示初始状态）
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt平方根 recip倒数 在某些逆扩散公式中用于对噪声做归一化处理
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# 逆扩散过程中的后验方差
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


@torch.no_grad()
def sample_timestep(model, x, t):
    """
    返回去噪后的xt图片
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # 获得xt
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        #最后一张图片不需要增置额外的噪声
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, device, img_size, T):
    # # 随机生成噪声图片
    # img = torch.randn((1, 3, img_size, img_size), device=device)

    # 使用真实图片
    img = plt.imread("noising_last_image.png")
    plt.imshow(img)
    plt.show()
    data_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),  # numpy --> PIL
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )
    img = data_transforms(img).unsqueeze(0).to(device)
    if img.shape[1] == 4:  # 检查是否有透明通道
        img = img[:, :3, :, :]  # 保留前三个通道

    show_tensor_image(img, is_show=True)

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    stepsize = int(T / num_images)

    # Reversed iteration
    for i in reversed(range(0, T)):
        t = torch.tensor([i], device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        img = torch.clamp(img, -1.0, 1.0)  # 本来就标准化到【-1， 1】了
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(num_images - i / stepsize))
            show_tensor_image(img.detach().cpu(), is_show=True)
    plt.savefig("sample.png")


if __name__ == "__main__":
    img_size = 64
    model = SimpleUnet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载预训练模型
    model.load_state_dict(torch.load("pretrain_models/ddpm_mse_epochs_10.pth"))
    model.to(device)
    sample_plot_image(model=model, device=device, img_size=img_size, T=T)