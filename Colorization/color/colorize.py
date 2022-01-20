from color.model import Colorization
from pathlib import Path
import torch
import torchvision
from torchvision import models
from loguru import logger
import cv2
import numpy as np
from color.train import save_image, concatente_and_colorize


def colorize(input_img, output_img):
    path: Path = sorted([x for x in Path('models').iterdir() if x.is_dir()], reverse=True)[0]
    path = sorted(path.iterdir(), reverse=True)[0]
    checkpoint = torch.load(path.open('rb'), map_location='cuda:0')
    model = Colorization(256).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])

    img = cv2.imread(input_img)
    img = img.astype(np.float32)
    img /= 255.0

    # *** Resize the color image to pass to encoder ***
    img = cv2.resize(img, ((img.shape[1] + 7) // 8 * 8, (img.shape[0] + 7) // 8 * 8))

    ''' Encoder Images '''
    # *** Convert the encoder color image to normalized lab space ***
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # *** Splitting the lab images into l-channel, a-channel, b-channel ***
    l_img = lab_img[:, :, 0]

    # *** Normalizing l-channel between [-1,1] ***
    l_img = l_img / 50.0 - 1.0

    # *** Repeat the l-channel to 3 dimensions ***
    l_img = torchvision.transforms.ToTensor()(l_img)
    l_img = l_img.expand(3, -1, -1)

    l_img = l_img.cuda()[None, :]

    resnet_model = models.resnet50(pretrained=True, progress=True).float().cuda()
    resnet_model.eval()
    resnet_model = resnet_model.float()

    # *** Intialize Model to Eval Mode ***
    model.eval()

    # *** Forward Propagation ***
    img_embs = resnet_model(l_img.float())
    output_ab = model(l_img, img_embs)
    logger.info(l_img.shape)
    logger.info(img_embs.shape)

    # *** Adding l channel to ab channels ***
    color_img = concatente_and_colorize(torch.stack([l_img[:, 0, :, :]], dim=1), output_ab)

    save_image(color_img[0], output_img)
