import numpy as np
import cv2
from os import makedirs
from os.path import join as path_join 
from pathlib import Path
from torch import Tensor, save as torch_save
from models import FacenetPipeline
from typing import List

class Input():
  
  def __init__(self, directory:str = "datasets"):
    
    self.directory = Path(directory)
    self.__ensure_dir__()
    
  def __ensure_dir__(self):
    # Check if the path exists or not
    self.directory.mkdir(parents=True, exist_ok=True)

  def __receive_username__(self):
    label= input("Enter label name:")
    return label
  
  def __start_capture__(self):
    
    model = FacenetPipeline()
    
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    curr = 0
    arr = []

    while(True): 
      
      _, frame = vid.read() 
      # cv2.imshow('Capturing sample face for inference', frame) 
      
      print(frame)
      if frame is not None:
        cropped_frame = cv2.resize(frame, (80, 80))

        result = model(cropped_frame)
        print(result)
        if result is not None:
          image, _ = result
          arr.append(image)
          curr = curr + 1
      
      if cv2.waitKey(1) & 0xFF == ord('q') or curr >= 10: 
          break

    # Release the video
    vid.release() 
    cv2.destroyAllWindows()
    
    del model
    return arr

  def __serialize_tensor_to_file__(self, name: str, tensors: List[Tensor]):
    _pathname = path_join(self.directory.name, name)
    
    # If not exists, make it
    makedirs(_pathname, exist_ok=True)
    
    for idx, tensor in enumerate(tensors):
      filename = path_join(_pathname, "{}.pt".format(idx))
      torch_save(tensor, filename)
    
  def start(self):
    # Self username
    username = self.__receive_username__()
    arr = self.__start_capture__()
    
    self.__serialize_tensor_to_file__(username, arr)

_input = Input()
_input.start()