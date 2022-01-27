from pathlib import Path

import click
import cv2
from loguru import logger
import numpy as np
from torchvision import models
from torchvision import transforms as T
from torchvision.utils import save_image
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import wandb


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        # 4, 9, 16, 23, 30
        vgg_layers = list(vgg.features.children())
        self.block_1 = torch.nn.Sequential(*vgg_layers[:4])
        self.block_2 = torch.nn.Sequential(*vgg_layers[4:9])
        self.block_3 = torch.nn.Sequential(*vgg_layers[9:16])
        self.block_4 = torch.nn.Sequential(*vgg_layers[16:23])
        self.block_5 = torch.nn.Sequential(*vgg_layers[23:30])

    def forward(self, x):
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        return x_1, x_2


@click.command()
@click.argument('from_img_path', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('to_img_path', type=click.Path(exists=True, dir_okay=False, readable=True))
def morph(from_img_path: Path, to_img_path: Path):
    wandb.init(project='Morph', entity='sphericalpotatoinvacuum')

    from_img = cv2.cvtColor(cv2.imread(from_img_path), cv2.COLOR_BGR2RGB)
    to_img = cv2.cvtColor(cv2.imread(to_img_path), cv2.COLOR_BGR2RGB)
    from_img = cv2.resize(from_img, (to_img.shape[1], to_img.shape[0]))
    save_path = Path('outputs') / datetime.datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S')
    save_path.mkdir(parents=True)

    tt = T.ToTensor()
    from_img = tt(from_img).cuda().unsqueeze(0)
    to_img = tt(to_img).cuda().unsqueeze(0)

    save_image(from_img, save_path / '0.png')

    from_img.requires_grad = True
    model = Model().eval().cuda()
    optimizer = torch.optim.Adam([from_img], 0.02)
    torch_lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=10)

    point_epochs = 10

    ns = list(map(lambda x: np.prod(x[0].shape), model(from_img)))

    avg_loss = 0.0
    last_loss = 10

    for epoch in range(1, 2001):
        optimizer.zero_grad()

        from_embs = model(from_img)
        to_embs = model(to_img)

        loss = sum([torch.norm(from_emb - to_emb) ** 2 / n for from_emb, to_emb, n in zip(from_embs, to_embs, ns)])
        loss.backward()

        optimizer.step()
        torch_lr_scheduler.step(loss.item())

        avg_loss += loss.item()
        wandb.log({"Loss": loss.item()})

        if epoch % point_epochs == 0:
            avg_loss /= point_epochs
            logger.info(f'Epoch: {epoch}. Average loss of last {point_epochs} epochs: {avg_loss}')
            save_image(from_img[0], save_path / f'{epoch // point_epochs}.png')

            last_loss = avg_loss
            avg_loss = 0


if __name__ == '__main__':
    morph()
