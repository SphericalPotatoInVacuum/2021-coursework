import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models.vgg import vgg11, vgg16, vgg19
from tqdm import tqdm
from pathlib import Path
import datetime


def train_one_epoch(model, train_dataloader, criterion, optimizer, device="cuda:0"):
    progress_bar = tqdm(train_dataloader)
    model = model.to(device).train()
    idx = 0
    for (images, labels) in progress_bar:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if idx % 10 == 0:
            progress_bar.set_description("Loss = {:.4f}".format(loss.item()))
        idx += 1


def predict(model, val_dataloader, criterion, device="cuda:0"):
    model.eval()

    losses = torch.Tensor()
    predicted_classes = torch.Tensor()
    true_classes = torch.Tensor()
    cumulative_loss = 0
    acc = 0
    model = model.to(device).eval()
    with torch.no_grad():
        for images, labels in tqdm(val_dataloader):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            cumulative_loss += loss.item()
            acc += (preds.argmax(1) == labels).float().mean()
    return losses, predicted_classes, true_classes


def train():
    model_save_path = Path('models') / datetime.datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S')
    model_save_path.mkdir(parents=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize([0.4803, 0.4481, 0.3976], [0.2769, 0.2690, 0.2820]),
    ])
    train_dataset = ImageFolder('data/Typefaces', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=15, shuffle=True)

    val_dataset = ImageFolder('data/Typefaces', transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=15, shuffle=True)

    model = vgg19(pretrained=True).cuda()
    model.classifier[6] = torch.nn.Linear(4096, len(train_dataset.classes))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    n_epochs = 5

    model.to(device)
    for epoch in range(n_epochs):
        train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        losses, predicted_classes, true_classes = predict(model, val_dataloader, criterion, device)
        if scheduler is not None:
            scheduler.step(losses.mean())
        checkpoint = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict(),
                      'val_loss': losses.mean()}
        torch.save(checkpoint, model_save_path / str(epoch + 1))
