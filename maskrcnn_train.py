import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import yaml
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../Desktop/")
print(ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib

# Directory to save logs and trained models
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class FruitConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "fruit"
    # backbone
    BACKBONE = "resnet50"
    # gpu setting
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 class
    # images size
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # anchor scale
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)

    STEPS_PER_EPOCH = 60


class FruitDataset(utils.Dataset):
    # Get how many instances (objects) are in the picture
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # Parse the yaml file obtained in labelme to get the instance label corresponding to each layer of mask
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.full_load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    # Rewrite draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        # print("info-->",info)
        # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    # Rewrite load_shapes, which contains its own categories
    # And added path, mask_path, yaml_path to self.image_info
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_fruit(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("fruit", 1, "apple")
        self.add_class("fruit", 2, "lemon")
        # self.add_class("shapes", 2, "tool2")
        for i in range(count):
            # 获取图片宽和高
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = "{}/label_{}.png".format(mask_floder, i)
            yaml_path = "{}/info_{}.yaml".format(yaml_floder, i)
            cv_img = cv2.imread("{}/img_{}.png".format(img_floder, i))
            self.add_image("fruit", image_id=i, path="{}/img_{}.png".format(img_floder, i),
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    # rewrite load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("Apple") != -1:
                # print "apple"
                labels_form.append("apple")
            elif labels[i].find("lemon") != -1:
                # print "lemon"
                labels_form.append("lemon")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)





def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# Basic Settings
dataset_root_path = ROOT_DIR + "/applemon_dataset"
img_floder = os.path.join(dataset_root_path, "pic")
mask_floder = os.path.join(dataset_root_path, "cv2_mask")
yaml_floder = os.path.join(dataset_root_path, "yaml")
# yaml_floder = dataset_root_path
imglist = os.listdir(img_floder)
count = len(imglist)
# os.listdir(mask_floder)

# train and val dataset preparation
dataset_train = FruitDataset()
dataset_train.load_fruit(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_train.prepare()

# print("dataset_train-->",dataset_train._image_ids)

dataset_val = FruitDataset()
dataset_val.load_fruit(10, img_floder, mask_floder, imglist, dataset_root_path)
dataset_val.prepare()

# use data augmentation
augmentation = iaa.SomeOf((0, 2), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
               iaa.Affine(rotate=180),
               iaa.Affine(rotate=270)]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])


# Create models in training mode
config = FruitConfig()
config.display()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last models you trained and continue training
    checkpoint_file = model.find_last()
    model.load_weights("/home/hqq/logs/fruit20200223T1502/mask_rcnn_fruit_0010.h5", by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
           learning_rate=config.LEARNING_RATE,
           epochs=10,
           augmentation=augmentation,
           layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            augmentation=augmentation,
            layers="all")
