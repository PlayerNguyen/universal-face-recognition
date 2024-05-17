from os.path import join as path_join
from os import listdir as ls
from torch import load as torch_load
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Sampler


class FeaturesDataset(Dataset):
  def __init__(self, dirname: str = 'datasets') -> None:
    super().__init__()
    
    self.path = Path(dirname)
  
    # Read all items
    self.label = ls(self.path)
    
  def __len__(self):
    return len(self.label)
  
  def __getitem__(self, idx):
    if idx > len(self.label):
      raise("Cannot found the data with idx {}".format(idx))
    
    label = self.label[idx]
    items = ls(path_join(self.path, label))
    
    arr = []
    for item_name in items:
      item_file_path = path_join(self.path, label, item_name)
      features = torch_load(item_file_path)
      arr.append(features)
    
    return arr, label  