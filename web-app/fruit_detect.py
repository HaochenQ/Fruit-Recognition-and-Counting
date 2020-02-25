import os
import sys
import random
import math
import time
from timeit import default_timer as timer
import ax
import cv2
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
# !{sys.executable} -m pip install model
# import utils
from PIL import Image, ImageFont, ImageDraw
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import yolo
from yolo import YOLO

# Root directory of the project
ROOT_DIR = os.getcwd()
print(ROOT_DIR)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, "./maskrcnn_models/mask_rcnn_fruit_0010.h5")  # 这里输入模型权重的路径


class FruitConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fruit"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3 # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.4


class InferenceConfig(FruitConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
# config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)


# class name for mask r-cnn model
class_names = ['BG','apple','lemon','pear']

# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'lemon', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']



def mask_detect(imagePath):
    try:
        image = skimage.io.imread(imagePath, pilmode="RGB")
    except (IOError, SyntaxError) as e:
        print('Bad file:', imagePath);

    # print(image)
    # Run detection
    results = model.detect([image], verbose=1)
    # print(results)
    # Visualize results
    r = results[0]
    # print(r)
    # print(r['rois'])
    print(r['rois'].shape[0])
    # print('masks:'+r['masks'])
    # print('class_ids:'+r['class_ids'])
    # print(r['scores'])
    '''
    change visualize.py in maskrcnn package line 169:
        if auto_show:
        plt.savefig('./maskedImages/result.png', bbox_inches='tight', pad_inches=-0.5, orientation='landscape')
        # plt.show()

    '''
    start = timer()
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    end =timer()
    timee = end -start
    print("maskrcnn processing time:"+str(timee))
    # visualize.draw_boxes(image)
    return r['rois'].shape[0]

def yolo_detect(img_path):
    count = yolo.yolo_detect_img(img_path)
    return count