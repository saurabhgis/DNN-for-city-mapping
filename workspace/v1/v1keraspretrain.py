from keras.applications import xception,vgg16,vgg19,resnet50,inception_v3,inception_resnet_v2,mobilenet,mobilenet_v2,densenet,nasnet
import numpy as np

#Load the Xception model
Xcep_model = xception.Xception(weights='imagenet')

# # Load the VGG16 model
# vgg16_model = vgg16.VGG16(weights='imagenet')
#
# #Load the VGG19 model
# vgg19_model = vgg19.VGG19(weights = 'imagenet')
#
# # Load the Inception_V3 model
# inception_model = inception_v3.InceptionV3(weights='imagenet')
#
# # Load the Inception_resner_v2 model
# inception_res = inception_resnet_v2.InceptionResNetV2(weights='imagenet')
#
# # Load the ResNet50 model
# resnet_model = resnet50.ResNet50(weights='imagenet')
#
# # Load the MobileNet model
# mobilenet_model = mobilenet.MobileNet(weights='imagenet')
#
# # Load the MobileNet_v2 model
# mobilenet_v2 = mobilenet_v2.MobileNetV2(weights='imagenet')
#
# # Load the DenseNet model
# densenet_model = densenet.DenseNet201(weights='imagenet')
#
# # Load the NASNet model
# nasnet_model = nasnet.NASNetLarge(weights='imagenet')

import google_streetview.api
import google_streetview.helpers

# 50.10291748018805, 14.39132777985096              dejvice
# 50.0795436,14.3907308                             Strahov
# 50.0746767,14.418974                              Karlovo namesti
apiargs = {
  'location': '50.10291748018805, 14.39132777985096 ',
  'size': '640x640',
  'heading': '0;90;180;270',
  'fov': '0;90:',
  'key': 'AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo',
  'pitch': '-90;0;90'
}

# Get a list of all possible queries from multiple parameters
api_list = google_streetview.helpers.api_list(apiargs)

# Create a results object for all possible queries
resultsg = google_streetview.api.results(api_list)

# Preview results
#resultsg.preview()

# Download images to directory 'downloads'
resultsg.download_links('StreetImages')

# Save metadata
resultsg.save_metadata('metadata.json')

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import os
import skimage.io
from skimage.transform import resize
from PIL import Image
#%matplotlib inline
ROOT_DIR = os.getcwd()
IMAGE_DIR = os.path.join(ROOT_DIR,'downloads')

# Load image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
procimg = np.empty(len(file_names), dtype=object)

# print(file_names)
for n in range(0, len(file_names)):
    print('     ')

    if file_names[n] == 'metadata.json':
        break;

    # Load the image
    image = Image.open(os.path.join(IMAGE_DIR, file_names[n]))

    # processing the Image
    image_resized = image.resize((224, 224), Image.ANTIALIAS)
    numpy_image = img_to_array(image_resized)
    image_batch = np.expand_dims(numpy_image, axis=0)

    # prepare the image for the VGG model
    processed_xception = xception.preprocess_input(image_batch.copy())
    processed_vgg16 = vgg16.preprocess_input(image_batch.copy())
    processed_vgg19 = vgg19.preprocess_input(image_batch.copy())
    processed_resnet = resnet50.preprocess_input(image_batch.copy())
    processed_inception = inception_v3.preprocess_input(image_batch.copy())
    processed_inception_resnet = inception_resnet_v2.preprocess_input(image_batch.copy())
    processed_mobilenet = mobilenet.preprocess_input(image_batch.copy())
    processed_densenet = densenet.preprocess_input(image_batch.copy())
    processed_nasnet = nasnet.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class (vgg16)
    #predictions = Xcep_model.predict(processed_xception)
    predictions = Xcep_model.predict(image_batch)
    # Display Image
    #   plt.imshow(image)
    #   plt.show()
    #   plt.imshow(image_resized)
    #   plt.show()
    #   plt.imshow(np.uint8(numpy_image))
    #   plt.show()
    #   plt.imshow(np.uint8(image_batch[0]))
    #   plt.show()
    plt.imshow(np.uint8(image_batch[0]))
    plt.show()
    # plt.imshow(np.uint8(processed_vgg16[0]))
    # plt.show()
    # plt.imshow(np.uint8(processed_vgg19[0]))
    # plt.show()
    # plt.imshow(np.uint8(processed_resnet[0]))
    # plt.show()
    # plt.imshow(np.uint8(processed_inception[0]))
    # plt.show()
    # plt.imshow(np.uint8(processed_inception_resnet[0]))
    # plt.show()
    # plt.imshow(np.uint8(processed_mobilenet[0]))
    # plt.show()
    # plt.imshow(np.uint8(processed_densenet[0]))
    # plt.show()
    # plt.imshow(np.uint8(processed_nasnet[0]))
    # plt.show()
    # # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    for prediction in decode_predictions(predictions)[0]:
        print(prediction[1], prediction[2])
        print('           ')
