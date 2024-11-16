import torch
import matplotlib.pyplot as plt
from dataloader import load_transformed_dataset, show_tensor_image


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, time_step, x_shape):
    """
    返回vals中对应的t的列表
    """
    batch_size = time_step.shape[0]
    out = vals.gather(-1, time_step.cpu())  # vals和time_step的维度需要相同
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(time_step.device)


def forward_diffusion_sample(x_0, time_step, device="cpu"):
    """
    返回增置噪声的图片
    """
    noise = torch.randn_like(x_0)
    # 批次中每张图片对应的时间的噪声比重序列
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, time_step, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, time_step, x_0.shape)
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(
        device
    ), noise.to(device)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


if __name__ == "__main__":
    # Simulate forward diffusion
    dataloader = load_transformed_dataset()
    image = next(iter(dataloader))[0]

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    stepsize = int(T / num_images)

    for idx in range(0, T, stepsize):
        time_step = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images + 1, int(idx / stepsize) + 1)
        img, noise = forward_diffusion_sample(image, time_step)
        show_tensor_image(img)
        if idx == T - stepsize:
            last_img = img
    plt.savefig("forward_nosing.png")
    plt.show()

    # 在循环结束后单独保存最后一张图像
    plt.figure()
    show_tensor_image(last_img)
    plt.axis("off")
    plt.savefig("noising_last_image.png")
    # plt.show()
