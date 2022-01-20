import time
import datetime
from pathlib import Path

import torch
from loguru import logger
from torchvision import models
from torch.utils.data import Dataset

from color.config import Config
from color.model import Colorization
import torchvision
import cv2
import numpy as np
from torchvision.utils import save_image


class CustomDataset(Dataset):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.files = list(root_dir.iterdir())
        logger.info(f'File[0]: {self.files[0]}, Total Files: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            # *** Read the image from file ***
            self.rgb_img = cv2.imread(str(self.files[index]))

            if self.rgb_img is None:
                raise Exception

            self.rgb_img = self.rgb_img.astype(np.float32)
            self.rgb_img /= 255.0

            # *** Resize the color image to pass to encoder ***
            rgb_encoder_img = cv2.resize(self.rgb_img, (224, 224))

            # *** Resize the color image to pass to decoder ***
            rgb_resnet_img = cv2.resize(self.rgb_img, (300, 300))

            ''' Encoder Images '''
            # *** Convert the encoder color image to normalized lab space ***
            self.lab_encoder_img = cv2.cvtColor(rgb_encoder_img, cv2.COLOR_BGR2Lab)

            # *** Splitting the lab images into l-channel, a-channel, b-channel ***
            l_encoder_img = self.lab_encoder_img[:, :, 0]
            a_encoder_img = self.lab_encoder_img[:, :, 1]
            b_encoder_img = self.lab_encoder_img[:, :, 2]

            # *** Normalizing l-channel between [-1,1] ***
            l_encoder_img = l_encoder_img / 50.0 - 1.0

            # *** Repeat the l-channel to 3 dimensions ***
            l_encoder_img = torchvision.transforms.ToTensor()(l_encoder_img)
            l_encoder_img = l_encoder_img.expand(3, -1, -1)

            # *** Normalize a and b channels and concatenate ***
            a_encoder_img = (a_encoder_img / 128.0)
            b_encoder_img = (b_encoder_img / 128.0)
            a_encoder_img = torch.stack([torch.Tensor(a_encoder_img)])
            b_encoder_img = torch.stack([torch.Tensor(b_encoder_img)])
            ab_encoder_img = torch.cat([a_encoder_img, b_encoder_img], dim=0)

            ''' Inception Images '''
            # *** Convert the resnet color image to lab space ***
            self.lab_resnet_img = cv2.cvtColor(rgb_resnet_img, cv2.COLOR_BGR2Lab)

            # *** Extract the l-channel of resnet lab image ***
            l_resnet_img = self.lab_resnet_img[:, :, 0] / 50.0 - 1.0

            # *** Convert the resnet l-image to torch Tensor and stack it in 3 channels ***
            l_resnet_img = torchvision.transforms.ToTensor()(l_resnet_img)
            l_resnet_img = l_resnet_img.expand(3, -1, -1)

            ''' return images to data-loader '''
            rgb_encoder_img = torchvision.transforms.ToTensor()(rgb_encoder_img)
            return l_encoder_img, ab_encoder_img, l_resnet_img, rgb_encoder_img, str(self.files[index].name)

        except Exception as e:
            logger.error(f'Exception at {self.files[index]}, {e}')
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), 'Error'


# Convert Tensor Image -> Numpy Image -> Color  Image -> Tensor Image
def concatente_and_colorize(im_lab, img_ab):
    logger.info(f'{im_lab.shape}, {img_ab.shape}')
    # Assumption is that im_lab is of size [1,3,224,224]
    # print(im_lab.size(),img_ab.size())
    np_img = im_lab[0].cpu().detach().numpy().transpose(1, 2, 0)
    lab = np.empty([*np_img.shape[0:2], 3], dtype=np.float32)
    lab[:, :, 0] = np.squeeze(((np_img + 1) * 50))
    lab[:, :, 1:] = img_ab[0].cpu().detach().numpy().transpose(1, 2, 0) * 127
    np_img = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    color_im = torch.stack([torchvision.transforms.ToTensor()(np_img)], dim=0)
    return color_im


def train(config: Config):
    if config.wb_enabled:
        import wandb

        wandb.init(project=config.project_name, entity=config.entity)

    resnet_model = models.resnet50(pretrained=True, progress=True).float().cuda()
    resnet_model.eval()
    resnet_model = resnet_model.float()

    loss_criterion = torch.nn.MSELoss(reduction='mean').cuda()
    milestone_list = list(range(0, config.total_epoch, 5))
    model = Colorization(256).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestone_list,
        gamma=config.lr_decay
    )

    train_dataset = CustomDataset(config.data_path / 'train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8
    )

    validataion_dataset = CustomDataset(config.data_path / 'val')
    validation_dataloader = torch.utils.data.DataLoader(
        validataion_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=8
    )

    model_save_path = Path('models') / datetime.datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S')
    model_save_path.mkdir()

    logger.info(
        f'Train: {len(train_dataloader)}, '
        f'Total Images: {len(train_dataloader) * config.batch_size}'
    )
    logger.info(
        f'Valid: {len(validation_dataloader)}, '
        f'Total Images: {len(validation_dataloader) * config.batch_size}'
    )

    for epoch in range(config.total_epoch):
        logger.info(f'Starting epoch #{epoch + 1}')

        # *** Training step ***
        loop_start = time.time()
        avg_loss = 0.0
        batch_loss = 0.0
        main_start = time.time()
        model.train()

        for idx, (img_l_encoder, img_ab_encoder, img_l_resnet, img_rgb, file_name) in enumerate(train_dataloader):
            # *** Skip bad data ***
            if not img_l_encoder.ndim:
                continue

            # *** Move data to GPU if available ***
            img_l_encoder = img_l_encoder.cuda()
            img_ab_encoder = img_ab_encoder.cuda()
            img_l_resnet = img_l_resnet.cuda()

            # *** Initialize Optimizer ***
            optimizer.zero_grad()

            # *** Forward Propagation ***
            img_embs = resnet_model(img_l_resnet.float())
            output_ab = model(img_l_encoder, img_embs)

            # *** Back propogation ***
            loss = loss_criterion(output_ab, img_ab_encoder.float())
            loss.backward()

            # *** Weight Update ****
            optimizer.step()

            # *** Reduce Learning Rate ***
            scheduler.step()

            # *** Loss Calculation ***
            avg_loss += loss.item()
            batch_loss += loss.item()

            # *** Print stats after every point_batches ***
            if (idx + 1) % config.point_batches == 0:
                loop_end = time.time()
                logger.info(
                    f'Batch: {idx + 1}, '
                    f'Batch time: {loop_end - loop_start: 3.1f}s, '
                    f'Batch Loss: {batch_loss / config.point_batches:7.5f}'
                )
                loop_start = time.time()
                batch_loss = 0.0

        # *** Print Training Data Stats ***
        train_loss = avg_loss / len(train_dataloader) * config.batch_size
        logger.info(f'Training Loss: {train_loss}, Processed in {time.time() - main_start:.3f}s')

        # *** Validation Step ***
        avg_loss = 0.0
        loop_start = time.time()
        # *** Intialize Model to Eval Mode for validation ***
        model.eval()
        for idx, (img_l_encoder, img_ab_encoder, img_l_resnet, img_rgb, file_name) in enumerate(validation_dataloader):
            # *** Skip bad data ***
            if not img_l_encoder.ndim:
                continue

            # *** Move data to GPU if available ***
            img_l_encoder = img_l_encoder.cuda()
            img_ab_encoder = img_ab_encoder.cuda()
            img_l_resnet = img_l_resnet.cuda()

            # *** Forward Propagation ***
            img_embs = resnet_model(img_l_resnet.float())
            output_ab = model(img_l_encoder, img_embs)

            # *** Loss Calculation ***
            loss = loss_criterion(output_ab, img_ab_encoder.float())
            avg_loss += loss.item()

        val_loss = avg_loss / len(validation_dataloader) * config.batch_size
        logger.info(f'Validation Loss: {val_loss}, Processed in {time.time() - loop_start:.3f}s')

        logger.success(
            f'Finished epoch #{epoch + 1} out of {config.total_epoch}. '
            f'Train loss: {train_loss:.5f}, val loss: {val_loss:.5f}'
        )
        if config.wb_enabled:
            wandb.log({"Validation loss": val_loss, "Train loss": train_loss}, step=epoch)

        # *** Save the Model to disk ***
        checkpoint = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict(),
                      'train_loss': train_loss, 'val_loss': val_loss}
        torch.save(checkpoint, model_save_path / str(epoch + 1))
        logger.info(f'Model saved at: {model_save_path / str(epoch + 1)}')

    # ### Inference

    test_dataset = CustomDataset(config.data_path / 'test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    logger.info(f'Test: {len(test_dataloader)} Total Image: {len(test_dataloader)}')

    # *** Inference Step ***
    avg_loss = 0.0
    loop_start = time.time()
    batch_start = time.time()
    batch_loss = 0.0

    for idx, (img_l_encoder, img_ab_encoder, img_l_resnet, img_rgb, file_name) in enumerate(test_dataloader):
        # *** Skip bad data ***
        if not img_l_encoder.ndim:
            continue

        # *** Move data to GPU if available ***
        img_l_encoder = img_l_encoder.cuda()
        img_ab_encoder = img_ab_encoder.cuda()
        img_l_resnet = img_l_resnet.cuda()

        # *** Intialize Model to Eval Mode ***
        model.eval()

        # *** Forward Propagation ***
        img_embs = resnet_model(img_l_resnet.float())
        output_ab = model(img_l_encoder, img_embs)

        # *** Adding l channel to ab channels ***
        color_img = concatente_and_colorize(torch.stack([img_l_encoder[:, 0, :, :]], dim=1), output_ab)

        save_path = Path('outputs')
        save_path.mkdir(exist_ok=True)
        save_path /= file_name[0]
        save_image(color_img[0], save_path)

        # *** Printing to Tensor Board ***
        # grid = torchvision.utils.make_grid(color_img)
        # writer.add_image('Output Lab Images', grid, 0)

        # *** Loss Calculation ***
        loss = loss_criterion(output_ab, img_ab_encoder.float())
        avg_loss += loss.item()
        batch_loss += loss.item()

        if (idx + 1) % config.point_batches == 0:
            batch_end = time.time()
            logger.info(
                f'Batch: {idx + 1}, '
                f'Processing time for {config.point_batches}: {batch_end - batch_start:.3f}s, '
                f'Batch Loss: {batch_loss / config.point_batches}'
            )
            batch_start = time.time()
            batch_loss = 0.0

    test_loss = avg_loss / len(test_dataloader)
    logger.info(f'Test Loss: {test_loss} Processed in {time.time() - loop_start:.3f}s')
    # writer.close()
