# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
from pathlib import Path
from typing import List, Tuple

import scipy.io
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from loguru import logger
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm

from reid.config import Config
from reid.evaluate_gpu import evaluate_model
from reid.model import ResNet


@logger.catch
def test(config: Config):
    ######################################################################
    # Options
    # --------

    # which_epoch = opt.which_epoch
    name = config.model_name

    gpu_ids = [0]

    ms = config.scale

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    ######################################################################
    # Load Data
    # ---------
    #
    # We will use torchvision and torch.utils.data packages for loading the
    # data.
    #
    h, w = 256, 128

    data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = config.data_path / '../../interim'

    if config.multi:
        image_datasets = {x: datasets.ImageFolder(data_dir / x, data_transforms)
                          for x in ['gallery', 'query', 'multi-query']}
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x],
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=16) for x in [
                'gallery',
                'query',
                'multi-query']}
    else:
        image_datasets = {
            x: datasets.ImageFolder(
                data_dir / x,
                data_transforms) for x in [
                'gallery',
                'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config.batch_size,
                                                      shuffle=False, num_workers=8) for x in ['gallery', 'query']}
    use_gpu = torch.cuda.is_available()

    ######################################################################
    # Load model
    # ---------------------------

    def load_network(network):
        save_path = Path('models') / name / 'net_last.pth'
        network.load_state_dict(torch.load(save_path))
        return network

    ######################################################################
    # Extract feature
    # ----------------------
    #
    # Extract feature from  a trained model.
    #

    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature(model, dataloaders):
        features = torch.FloatTensor()
        count = 0
        for data in tqdm(dataloaders):
            img, label = data
            n, c, h, w = img.size()
            count += n
            ff = torch.FloatTensor(n, 512).zero_().cuda()

            for i in range(2):
                if(i == 1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
                for scale in ms:
                    if scale != 1:
                        # bicubic is only  available in pytorch>= 1.1
                        input_img = nn.functional.interpolate(
                            input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                    outputs = model(input_img)
                    ff += outputs
            # norm feature
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff.data.cpu()), 0)
        return features

    def get_id(img_path: List[Tuple[str, int]]):
        camera_id = []
        labels = []
        for path, v in img_path:
            # filename = path.split('/')[-1]
            filename = Path(path).name
            label = filename[0:4]
            camera = filename.split('c')[1]
            if label[0:2] == '-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels

    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    gallery_cam, gallery_label = get_id(gallery_path)
    query_cam, query_label = get_id(query_path)

    if config.multi:
        mquery_path = image_datasets['multi-query'].imgs
        mquery_cam, mquery_label = get_id(mquery_path)

    ######################################################################
    # Load Collected data Trained model
    model_structure = ResNet()

    # if opt.fp16:
    #    model_structure = network_to_half(model_structure)

    logger.info('Loading network')
    model = load_network(model_structure)
    logger.info('Loaded network')

    model.classifier.classifier = nn.Sequential()

    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    with torch.no_grad():
        gallery_feature = extract_feature(model, dataloaders['gallery'])
        query_feature = extract_feature(model, dataloaders['query'])
        if config.multi:
            mquery_feature = extract_feature(model, dataloaders['multi-query'])

    # Save to Matlab for check
    result = {
        'gallery_f': gallery_feature.numpy(),
        'gallery_label': gallery_label,
        'gallery_cam': gallery_cam,
        'query_f': query_feature.numpy(),
        'query_label': query_label,
        'query_cam': query_cam}
    scipy.io.savemat('pytorch_result.mat', result)

    logger.info('Evaluating')
    evaluate_model()

    if config.multi:
        result = {'mquery_f': mquery_feature.numpy(), 'mquery_label': mquery_label, 'mquery_cam': mquery_cam}
        scipy.io.savemat('multi_query.mat', result)
