import cv2
import numpy as np


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
        print(np.mean(mask))
        image = apply_mask(image, mask, color)
        # image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


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
    from mrcnn.config import Config
    from os.path import isfile, join
    import time
    '''
                    '50.0753397,14.4189888 ;'
                    '50.0795436,14.3907308 ;'
                    '50.1005043,14.3915658;'
                    '50.10291748018805, 14.39132777985096 ;'
    '''
    start = time.time()
    ROOT_DIR = os.getcwd()
    COCO_DIR = os.path.join(os.path.normpath(os.getcwd() + os.sep + os.pardir),'Mask RCNN' )
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(COCO_DIR,"mask_rcnn_coco.h5")

    if os.path.isdir('Detection') is False:
        os.mkdir('Detection')
    DET_DIR = os.path.join(ROOT_DIR, "Detection")
    DET_PATH = DET_DIR + os.sep
    #print(DET_PATH)
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