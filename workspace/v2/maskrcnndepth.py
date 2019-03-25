import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt

from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *
from os import listdir
from os.path import isfile, join

ROOT_DIR = os.getcwd()
IMAGE_PATH = '/media/varun/DATA/Projects/DNN for city mapping/workspace/v2/Detection'
CHECKPOINT_PATH = '/media/varun/DATA/Projects/DNN for city mapping/workspace/monodepth/models/model_cityscapes/model_cityscapes.data-00000-of-00001'
parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--encoder',          type=str,   help='type of encoder, vgg or resnet50', default='vgg')
#parser.add_argument('--image_path',       type=str,   help='path to the image', required=True)
#parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', required=True)
parser.add_argument('--image_path',       type=str,   help='path to the image', default=IMAGE_PATH)
parser.add_argument('--checkpoint_path',  type=str,   help='path to a specific checkpoint to load', default=CHECKPOINT_PATH)
parser.add_argument('--input_height',     type=int,   help='input height', default=256)
parser.add_argument('--input_width',      type=int,   help='input width', default=512)

args = parser.parse_args()

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image



def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def test_simple(params):
    """Test function."""

    left  = tf.placeholder(tf.float32, [2, args.input_height, args.input_width, 3])
    model = MonodepthModel(params, "test", left, None)
    onlyfiles =[]
    for f in listdir(IMAGE_PATH):
        if f is not 'metadat.json':
            onlyfiles.append(f)
    onlyfiles = [f for f in listdir(IMAGE_PATH) if isfile(join(IMAGE_PATH, f))]

    for n in range(0, len(onlyfiles) - 1):

        input_image = scipy.misc.imread(join(IMAGE_PATH, onlyfiles[n]), mode="RGB")
        original_height, original_width, num_channels = input_image.shape
        input_image = scipy.misc.imresize(input_image, [args.input_height, args.input_width], interp='lanczos')
        input_image = input_image.astype(np.float32) / 255
        input_images = np.stack((input_image, np.fliplr(input_image)), 0)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # RESTORE
        restore_path = args.checkpoint_path.split(".")[0]
        train_saver.restore(sess, restore_path)

        disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
        disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

        output_directory = os.path.join(ROOT_DIR,'output')
        output_name = onlyfiles[n]

        np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
        disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
        plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')

        print('done!')

def main(_):

    params = monodepth_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    test_simple(params)


if __name__ == '__main__':
    """
        test everything
    """
    import os
    import sys
    import coco
    import utils
    import model as modellib
    import google_streetview.api
    import google_streetview.helpers
    from os import listdir
    from os.path import isfile, join
    import time
    from mrcnn.config import Config

    '''
                    '50.0753397,14.4189888 ;'
                    '50.0795436,14.3907308 ;'
                    '50.1005043,14.3915658;'
                    '50.10291748018805, 14.39132777985096 ;'
    '''
    start = time.time()

    ROOT_DIR = os.getcwd()
    COCO_DIR = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir),'Mask RCNN')

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(COCO_DIR,"mask_rcnn_coco.h5")

    if os.path.isdir('Detection') is False:
        os.mkdir('Detection')

    DET_DIR = os.path.join(ROOT_DIR,'Detection')
    DET_PATH = DET_DIR + '/ '
    #DET_PATH = DET_DIR + '\ '

    apiargs = {
        'location': '50.0753397,14.4189888 ; 50.0795436,14.3907308 ;50.10291748018805, 14.39132777985096',
        'size': '640x640',
        'heading': '0;45;90;135;180;225;270',
        'fov': '90',
        'key': 'AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo',
        'pitch': '0'
    }

    # Get a list of all possible queries from multiple parameters
    api_list = google_streetview.helpers.api_list(apiargs)

    # Create a results object for all possible queries
    resultsg = google_streetview.api.results(api_list)

    # Preview results
    #resultsg.preview()

    # Download images to directory 'downloads'
    resultsg.download_links('Downloads')

    # Save metadata
    #resultsg.save_metadata('metadata.json')

    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class CocoConfig(Config):
        """Configuration for training on MS COCO.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
        # Give the configuration a recognizable name
        NAME = "coco"

        # We use a GPU with 12GB memory, which can fit two images.
        # Adjust down if you use a smaller GPU.
        IMAGES_PER_GPU = 2

        # Uncomment to train on 8 GPUs (default is 1)
        # GPU_COUNT = 8

        # Number of classes (including background)
        # NUM_CLASSES = 37  # COCO has 80 classes
        NUM_CLASSES = 81


    class InferenceConfig(CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # MAX_GT_INSTANCES = 100
        # TRAIN_ROIS_PER_IMAGE = 50
        # BACKBONE = "resnet50" #not working at all!
        # RPN_ANCHOR_STRIDE = 2
        POST_NMS_ROIS_TRAINING = 1000
        POST_NMS_ROIS_INFERENCE = 500
        IMAGE_MIN_DIM = 400  # really much faster but bad results
        IMAGE_MAX_DIM = 512
        # DETECTION_MAX_INSTANCES = 50 #a little faster but some instances not recognized

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    class_names = [
        'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]
    onlyfiles = []
    while True:
        images = os.path.join(ROOT_DIR,'Downloads')
        for f in listdir(images):
            if f is not 'metadat.json':
                onlyfiles.append(f)
        onlyfiles = [f for f in listdir(images) if isfile(join(images, f))]
        frame = np.empty(len(onlyfiles), dtype=object)
        for n in range(0, len(onlyfiles)-1):
            frame[n] = cv2.imread(join(images, onlyfiles[n]))
            results = model.detect([frame[n]], verbose=0)
            r = results[0]
            frame[n] = display_instances(
                frame[n], r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
            )
            cv2.imwrite(DET_PATH + str(n) + '.jpeg', frame[n])
        end = time.time()
        print(end-start)
        cv2.destroyAllWindows()
        break

    tf.app.run()