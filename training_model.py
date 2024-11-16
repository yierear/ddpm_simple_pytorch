import os
from forward_noising import forward_diffusion_sample, T
from unet import SimpleUnet
from dataloader import load_transformed_dataset
import torch.nn.functional as F
import torch
from torch.optim import Adam
import logging

logging.basicConfig(level=logging.INFO)


def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    # return F.l1_loss(noise_pred, noise)
    return F.mse_loss(noise_pred, noise)


if __name__ == "__main__":
    model = SimpleUnet()
    batch_size = 16
    epochs = 10
    model_path = os.path.join(os.getcwd(), 'pretrain_models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dataloader = load_transformed_dataset(batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for batch_idx, (batch, _) in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, T, (batch_size,), device=device).long()
            loss = get_loss(model, batch, t, device=device)
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                logging.info(f"Epoch {epoch} | Batch index {batch_idx:03d} Loss: {loss.item()}")

    torch.save(model.state_dict(), f"./pretrain_models/ddpm_mse_epochs_{epochs}.pth")
