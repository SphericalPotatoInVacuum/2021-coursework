from __future__ import print_function, division
from pytorch_metric_learning import losses, miners
from pathlib import Path
from reid.model import ResNet
from reid.config import Config
from loguru import logger
import time
import matplotlib.pyplot as plt
import rtoml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib


@logger.catch
def train(config: Config):
    matplotlib.use('agg')
    # from PIL import Image

    version = torch.__version__

    ######################################################################
    # Options
    # --------
    data_dir = config.data_path / '../../interim'
    name = config.model_name
    gpu_ids = [0]

    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    ######################################################################
    # Load Data
    # ---------
    #

    h, w = 256, 128

    transform_train_list = [
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Pad(10),
        transforms.RandomCrop((h, w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    transform_val_list = [
        transforms.Resize(size=(h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    if config.color_jitter:
        transform_train_list = [
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0)] + transform_train_list

    logger.info(f'Used transforms: {transform_train_list}')
    data_transforms = {
        'train': transforms.Compose(transform_train_list),
        'val': transforms.Compose(transform_val_list),
    }

    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(data_dir / 'train',
                                                   data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(data_dir / 'val',
                                                 data_transforms['val'])

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=config.batch_size,
                                                  shuffle=True, num_workers=8, pin_memory=True)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    use_gpu = torch.cuda.is_available()

    since = time.time()
    inputs, classes = next(iter(dataloaders['train']))
    logger.info(f'One dataloader iteration: {time.time() - since}s')

    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's write a general function to train a model. Here, we will
    # illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.

    y_loss = {}  # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []

    def fliplr(img):
        '''flip horizontal'''
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        # best_model_wts = model.state_dict()
        # best_acc = 0.0
        warm_up = 0.1  # We start from the 0.1*lrRate
        warm_iteration = round(dataset_sizes['train'] / config.batch_size) * \
            config.warm_epoch  # first 5 epoch

        miner = miners.TripletMarginMiner(margin=0.3, type_of_triplets='hard')
        criterion_triplet = losses.TripletMarginLoss(margin=0.3)

        # criterion_sphere = losses.SphereFaceLoss(751, embedding_size=512, margin=4)
        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch}/{num_epochs - 1}')

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                logger.info(f'Phase: {phase}')
                if phase == 'train':
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0
                # Iterate over data.
                for data in tqdm(dataloaders[phase]):
                    # get the inputs
                    inputs, labels = data
                    now_batch_size, c, h, w = inputs.shape
                    if now_batch_size < config.batch_size:  # skip the last batch
                        continue
                    # print(inputs.shape)
                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda().detach())
                        labels = Variable(labels.cuda().detach())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    # if we use low precision, input also need to be fp16
                    # if fp16:
                    #    inputs = inputs.half()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    if phase == 'val':
                        with torch.no_grad():
                            outputs = model(inputs)
                    else:
                        outputs = model(inputs)

                    """
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    """

                    logits, ff = outputs
                    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
                    ff = ff.div(fnorm.expand_as(ff))
                    loss = criterion(logits, labels)
                    _, preds = torch.max(logits.data, 1)
                    hard_pairs = miner(ff, labels)
                    loss += criterion_triplet(ff, labels, hard_pairs)  # /now_batch_size
                    # loss += criterion_sphere(ff, labels) / now_batch_size

                    del inputs
                    # use extra DG Dataset (https://github.com/NVlabs/DG-Net#dg-market)

                    # backward + optimize only if in training phase
                    if epoch < config.warm_epoch and phase == 'train':
                        warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                        loss = loss * warm_up

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    # statistics
                    if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                        running_loss += loss.item() * now_batch_size
                    else:  # for the old version like 0.3.0 and 0.3.1
                        running_loss += loss.data[0] * now_batch_size
                    del loss
                    running_corrects += float(torch.sum(preds == labels.data))

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0 - epoch_acc)
                # deep copy the model
                if phase == 'val':
                    last_model_wts = model.state_dict()
                    if epoch % 10 == 9:
                        save_network(model, epoch)
                    draw_curve(epoch)
                if phase == 'train':
                    scheduler.step()
            time_elapsed = time.time() - since
            logger.info(f'Training complete in {time_elapsed // 60:.0f}:{time_elapsed % 60:2.0f}')

        time_elapsed = time.time() - since
        logger.info(f'Training complete in {time_elapsed // 60:.0f}:{time_elapsed % 60:2.0f}')
        # print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(last_model_wts)
        save_network(model, 'last')
        return model

    ######################################################################
    # Draw Curve
    # ---------------------------
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")

    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig(Path('models') / name / 'train.jpg')

    ######################################################################
    # Save model
    # ---------------------------

    def save_network(network, epoch_label):
        save_filename = f'net_{epoch_label}.pth'
        save_path = Path('models') / name / save_filename
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrainied model and reset final fully connected layer.
    #

    model = ResNet(circle=True)

    # model to gpu
    model = model.cuda()

    optim_name = optim.SGD

    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    classifier_params = model.classifier.parameters()
    optimizer_ft = optim_name([
        {'params': base_params, 'lr': 0.1 * config.lr},
        {'params': classifier_params, 'lr': config.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config.total_epoch * 2 // 3, gamma=0.1)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 1-2 hours on GPU.
    #
    dir_name = Path('models') / name
    dir_name.mkdir(exist_ok=True)

    # save opts
    (dir_name / 'opts.toml').write_text(rtoml.dumps(config.dict(), pretty=True))

    criterion = nn.CrossEntropyLoss()

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=config.total_epoch)
