from dataset import FeaturesDataset

from models import FacenetPipeline
from PIL import Image
from torch import Tensor, cdist, device
import torchvision.transforms as transformer
import numpy as np
import cv2
from torch import device as torch_device
import torch.cuda as cuda

# Load torch as cuda
device = torch_device('cuda:0' if cuda.is_available() else 'cpu')

class Inference:
  
  def __init__(self, dirname: str = 'datasets'):
    self.dataset = FeaturesDataset(dirname)
    self.model = FacenetPipeline()
  
  def search_face(self, dataset: FeaturesDataset, image: Image.Image, threshold: float = 0.89):
    '''
    Search for any face inside the dataset.
    '''
    
    # Scale down image before 
    image = transformer.Resize((480, 480))(image)
    
    result = self.model(image)
    if result is None:
      return -1, None, None
    
    
    f, _ = result
    
    labels = []
    distances = []
    for data in dataset:
        features_map, label = data
        labels.append(label)
        arr = []
        for features in features_map:
          distance = cdist(f, features)
          arr.append(distance.item())
          
          # print(features.squeeze(0))
          
        distance = np.mean(arr)
        distances.append(distance)
    print(labels)
    print(distances)
    distances = Tensor(distances)
    # print(distances)
    min_idx = distances.argmin()
    min_idx= min_idx.item()
    return (-1, None, None) if distances[min_idx] > threshold else (min_idx, dataset.__getitem__(min_idx)[1], distances[min_idx])
  
  def start(self):
    
    # Load a datasets
    dataset = FeaturesDataset()
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # thread_pool = multiprocessing.Pool(4)
    
    while True:
      ret, frame = video.read(0)
      if ret is True:
        # cv2.imshow("Check attendance", frame)
        frame = cv2.resize(frame, (160, 160))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        
        result = self.search_face(dataset, image=image)
        print(result)
              
      
      if cv2.waitKey(1) & 0xFF == ord('q'): 
          break
    
    video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
  # inf = Inference()
  # value = inf.search_face()
  inf = Inference()
  inf.start()