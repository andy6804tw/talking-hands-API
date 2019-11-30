import cv2
import pickle as pkl
import imageio
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as TF
from face_alignment import FaceAlignment, LandmarksType

from app.modules.talkingHeads.utils import load_model, generate_image, plot_landmarks, generate_lm, image_to_video, generate_moving_video
import app.modules.talkingHeads.network as network

from flask import jsonify

# %load_ext autoreload
# %autoreload 2

def getTransform():
    G = network.Generator()
    G = load_model(G, "app/modules/talkingHeads/resource/han", "han")
    G = G.to("cuda:0")
    generate_moving_video(G, "app/static/source.mp4", "app/modules/talkingHeads/resource/han/han.npy", "app/static/result.mp4", "cuda:0")
    return jsonify({"code":200,"message": "轉換成功"})

def getResult():
    return 'ddon'

