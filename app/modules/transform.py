import cv2
import pickle as pkl
import imageio
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF
from face_alignment import FaceAlignment, LandmarksType

from app.modules.talkingHeads.utils import load_model, generate_image, plot_landmarks, generate_lm, image_to_video, generate_moving_video
import app.modules.talkingHeads.network as network

# %load_ext autoreload
# %autoreload 2

def getTransform():
  G = network.Generator()
  G = load_model(G, "app/modules/talkingHeads/resource/han", "han")
  G = G.to("cpu")
  generate_moving_video(G, "app/modules/talkingHeads/resource/demo.mov", "app/modules/talkingHeads/resource/han/han.npy", "app/static/test.mp4", "cpu")
  return 'ddon'

def getResult():
    return 'ddon'

