from dataset import FeaturesDataset
from torch.utils.data import DataLoader
from models import FacenetPipeline
from PIL import Image
import torch
import torchvision.transforms as transformer
import numpy as np

# dataset = FeaturesDataset()
# dataloader = DataLoader(dataset, batch_size=4)


img = Image.open('./sample/face.jpg')

# threshold = 0.8999999999

    