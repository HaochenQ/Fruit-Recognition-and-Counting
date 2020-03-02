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
MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, "./maskrcnn_models/mask_rcnn_fruit_0016.h5")  # 这里输入模型权重的路径
# MODEL_WEIGHT_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(MODEL_WEIGHT_PATH):
#     utils.download_trained_weights(MODEL_WEIGHT_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "pics")  # 这是输入你要预测图片的路径


class FruitConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fruit"
    BACKBONE = "resnet50"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2 # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 60

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.4


class InferenceConfig(FruitConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
# config.display()

# def get_ax(rows=1, cols=1, size=8):
#     """Return a Matplotlib Axes array to be used in
#     all visualizations in the notebook. Provide a
#     central point to control graph sizes.
#
#     Change the default size attribute to control the size
#     of rendered images
#     """
#     _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
#     return ax


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(MODEL_WEIGHT_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG','apple','lemon','pear']#['BG','lemon']

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


# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# for x in range(len(file_names)):
#     image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[x]))
#     # print(image)
#     # Run detection
#     results = model.detect([image], verbose=1)
#     # print(results)
#     # Visualize results
#     r = results[0]
#     # print(r)
#     # bbox:
#     print(r['class_ids'])
#     print(r['rois'].shape[0])
#     # print('masks:'+r['masks'])
#     # print('class_ids:'+r['class_ids'])
#     # print(r['scores'])
#     visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                                 class_names, r['scores'])

# plt.savefig('haha.png', bbox_inches='tight', pad_inches=-0.5, orientation='landscape')
# visualize.draw_boxes(image)


def mask_detect(imagePath):
    try:
        image = skimage.io.imread(imagePath, pilmode="RGB")
    except (IOError, SyntaxError) as e:
        print('Bad file:', imagePath);

    # print(image)
    # Run detection
    start = timer()
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

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    end =timer()
    timee = end -start
    print("maskrcnn processing time:"+str(timee))
    # visualize.draw_boxes(image)
    return r['rois'].shape[0]
# mask_detect("pics/lemon-citrus_1.jpg")
'''
YOLO PART

'''

'''def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (128, 255, 0), 4)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()

def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    # start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    # end = time.time()

    # print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)
        cv2.imwrite('maskedImages/result.png', image)

    return scores.shape[0]


# make_detection("./pics/lemon-citrus_1.jpg")
def yolo_detection(imagePath):
    yolo = YOLO(0.6, 0.5)
    all_classes=class_names[1:]
    image = cv2.imread(imagePath)
    number = detect_image(image, yolo, all_classes)
    return  number'''
# def yolo_detection(imagePath):
#     image = cv2.imread(imagePath)
#     r_image = YOLO.detect_image()
#     number = yolo_detection(image, yolo, all_classes)
#     return  number

def yolo_detect(img_path):
    # yolo = YOLO()
    # img_path = 'pics/lemon-citrus_1.jpg'
    # image = cv2.imread(img_path)
    count = yolo.yolo_detect_img(img_path)
    return count
