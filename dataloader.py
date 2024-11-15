from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import torch
import torchvision
import matplotlib.pyplot as plt

data_path = os.path.join(os.getcwd(), "stanford_cars")
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

def load_transformed_dataset(img_size=64, batch_size=32, is_train=True) -> DataLoader:
    # data transformations
    data_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    if os.path.exists(data_path):
        # 已下载
        train = torchvision.datasets.ImageFolder(root=train_path, transform=data_transforms)
        test = torchvision.datasets.ImageFolder(root=test_path, transform=data_transforms)
    else:
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        # 未下载
        train = torchvision.datasets.StanfordCars(root=train_path, download=True, transform=data_transforms)
        test = torchvision.datasets.StanfordCars(root=test_path, download=True, transform=data_transforms, split='test')

    # dataset = torch.utils.data.ConcatDataset([train, test])
    if is_train:
        dataset = train
    else:
        dataset = test

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def show_tensor_image(image, is_show=False):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),
            transforms.Lambda(lambda t: t * 225.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    if len(image.shape) == 4:
        image = image[0, :, :, :]

    plt.imshow(reverse_transforms(image))

    if is_show:
        plt.show()


def show_dataset(dataset, num_samples=20, cols=4):
    plt.figure(figsize=(15,15))
    for i, imgs in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols), cols, i+1)
        plt.imshow(imgs[0])
    plt.show()


if __name__ == '__main__':
    print(f'data path: {data_path}')
    data_loader = load_transformed_dataset(batch_size=32, is_train=True)
    print("show_tensor_image:")
    # for i, batch in enumerate(data_loader):
    #     inputs, labels = batch
    #     show_tensor_image(inputs, is_show=True)
    #
    #     if i == 2:
    #         break
    print("\nshow_dataset:")
    train = torchvision.datasets.ImageFolder(root=train_path)
    show_dataset(train, num_samples=20, cols=4)