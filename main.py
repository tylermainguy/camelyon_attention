import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

from glimpse_sensor import GlimpseSensor
from inception_pretrained import get_pretrained_inception


def main():

    # required preprocessing for inception
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # dataset = datasets.ImageFolder("data/train/downsampled", transform=transf)

    # train_loader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     batch_size=1,
    #     shuffle=True
    # )

    # iter_loader = iter(train_loader)

    # x, y = next(iter_loader)

    # location = torch.zeros(2, 1)

    # sensor = GlimpseSensor(256, 2)

    # patch = sensor.glimpse(x, location)

    # sq = torch.squeeze(patch)
    # sq = sq.permute(1, 2, 0)

    # plt.imshow(sq)
    # plt.show()

    get_pretrained_inception()


if __name__ == "__main__":
    main()
