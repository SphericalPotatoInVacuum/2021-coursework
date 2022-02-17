from torchvision.models import vgg11
from torch import nn

class TypefaceClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg11 = vgg11(pretrained=False)
        print(self.vgg11)
