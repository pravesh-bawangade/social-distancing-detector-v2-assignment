"""
@Author: Pravesh Bawangade
@filename: model.py
"""

import glob, os, tarfile, urllib
import tensorflow as tf
from utils import label_map_util
import numpy as np


class model:
    """
    Model class
    """
    def __init__(self, model_name):
        """
        initilization
        :param model_name: name of model to use.
        Example:
        'faster_rcnn_resnet50_coco_2018_01_28'
        'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
        'ssd_mobilenet_v1_coco_2018_01_28'
        """

        detection_graph, self.category_index = self.set_model(
            model_name,
            'mscoco_label_map.pbtxt')

        self.sess = tf.InteractiveSession(graph=detection_graph)
        self.image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
        self.detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
        self.detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
        self.detection_classes = detection_graph.get_tensor_by_name(
            "detection_classes:0"
        )
        self.num_detections = detection_graph.get_tensor_by_name("num_detections:0")

    @staticmethod
    def set_model(model_name, label_name):
        """
        Download model and return graph and category index
        :param model_name: Name of model to use.
        :param label_name: Label path
        :return: detection_graph, category_index
        """
        model_found = 0

        for file in glob.glob("*"):
            if file == model_name:
                model_found = 1

        # What model to download.
        model_name = model_name
        model_file = model_name + ".tar.gz"
        download_base = "http://download.tensorflow.org/models/object_detection/"

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        path_to_ckpt = model_name + "/frozen_inference_graph.pb"

        # List of the strings that is used to add correct label for each box.
        path_to_labels = os.path.join("data", label_name)

        num_classes = 90

        # Download Model if it has not been downloaded yet
        if model_found == 0:
            opener = urllib.request.URLopener()
            opener.retrieve(download_base + model_file, model_file)
            tar_file = tarfile.open(model_file)
            for file in tar_file.getmembers():
                file_name = os.path.basename(file.name)
                if "frozen_inference_graph.pb" in file_name:
                    tar_file.extract(file, os.getcwd())

        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        # Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I 		use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=num_classes, use_display_name=True
        )
        category_index = label_map_util.create_category_index(categories)

        return detection_graph, category_index

    def get_category_index(self):
        """
        returns category_index
        :return:
        """
        return self.category_index

    def detect(self, frame):
        """
        detect objects in given frame
        :param frame: input frame
        :return: scores, classes, boxes
        """
        # Actual detection.
        # input_frame = cv2.resize(frame, (350, 200))
        input_frame = frame

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(input_frame, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [
                self.detection_boxes,
                self.detection_scores,
                self.detection_classes,
                self.num_detections,
            ],
            feed_dict={self.image_tensor: image_np_expanded},
        )

        classes = np.squeeze(classes).astype(np.int32)
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        return scores, classes, boxes
