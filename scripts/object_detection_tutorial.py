#! /usr/bin/env python
import os
import pathlib

import numpy as np
import tensorflow as tf

from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2

import rospy
from amazon_ros_speech import talker
from sensor_msgs.msg import CompressedImage
from pointing_detection import img_utils

import time

already_processing = False





# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/home/bradsaund/research/tensorflow_model_zoo/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'

LAST_HUMAN_TIME = time.time()
GREET_DELAY_TIME_SEC = 60 * 10  # seconds


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    Image.fromarray(image_np).show("Processed")


def greet_new_people(output_dict):
    global LAST_HUMAN_TIME
    if is_person_in_image(output_dict):
        if GREET_DELAY_TIME_SEC < time.time() - LAST_HUMAN_TIME:
            print("Hi there new fellow")
            talker.say("Hello, I hope you have a great day!")
        LAST_HUMAN_TIME = time.time()


def is_person_in_image(output_dict):
    for score, label in zip(output_dict['detection_scores'], output_dict['detection_classes']):
        if score < 0.5:
            continue

        if category_index[label]['name'] != 'person':
            continue
        return True
    return False


def img_callback(img_msg):
    global already_processing
    if already_processing:
        print("skipping this call")
        return

    dt = (rospy.get_rostime() - img_msg.header.stamp)
    delay = dt.secs + dt.nsecs*1e-9
    if delay > 0.01:
        # print("Too far behind, skipping this call")
        return

    already_processing = True
    decompressed = img_utils.decompress_img(img_msg)
    decompressed = cv2.flip(decompressed, 1)
    t0 = time.time()
    output_dict = run_inference_for_single_image(detection_model, decompressed)
    vis_util.visualize_boxes_and_labels_on_image_array(
        decompressed,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    img_msg.data = img_utils.compress_img(decompressed)
    marked_pub.publish(img_msg)
    greet_new_people(output_dict)
    # print("Inference took {} seconds".format(time.time() - t0))

    already_processing = False


if __name__ == "__main__":
    rospy.init_node("object_detection")
    detection_model = load_model(model_name)
    talker.init()
    img_sub = rospy.Subscriber("/kinect2_victor_head/qhd/image_color/compressed", CompressedImage, img_callback,
                               queue_size=1)
    marked_pub = rospy.Publisher("/marked_image/compressed", CompressedImage, queue_size=1)
    rospy.spin()
