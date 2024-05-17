from torch import device as torch_device
import torch.cuda as cuda
from torch.nn import Module
from facenet_pytorch import MTCNN, InceptionResnetV1

class FacenetPipeline(Module):
  
  def __init__(self, training: bool = False, detect_threshold: float = 0.76):
    super().__init__()
    
    # Switch back device
    device = torch_device('cuda:0' if cuda.is_available() else 'cpu')
    
    
    # First stage, detect image
    self.detect = MTCNN(margin=32, device=device, post_process=True)
    # Second stage, recognition
    self.recognition = InceptionResnetV1(pretrained="vggface2", device=device)
    self.detect_threshold = detect_threshold
    
    if training is None or training is False:
      self.recognition.eval()
  
  def forward(self, image):
    image, prob = self.detect(image, return_prob=True)
    
    if self.detect_threshold is not None and prob is not None and prob < self.detect_threshold:
      return None
    
    # If cannot found face
    if image is None:
      return None
    
    # Extract features
    image = self.recognition(image.unsqueeze(0))
    return image, prob