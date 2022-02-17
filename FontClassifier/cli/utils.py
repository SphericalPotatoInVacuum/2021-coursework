from torch.utils.data import Dataset
from pathlib import Path
import cv2


class TypefaceDataset(Dataset):
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.images = []
        self.labels = []
        self.fonts = []
        for idx, font in enumerate(data_path.iterdir()):
            self.fonts.append(font.stem)
            for word in font.iterdir():
                for img in word.iterdir():
                    self.images.append(img)
                    self.labels.append(idx)
        self.class_num = len(self.labels)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.labels[idx]

    def __len__(self):
        return len(self.images)
