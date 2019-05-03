from styx_msgs.msg import TrafficLight

import os
import sys
import time
from PIL import Image

import numpy as np
import tensorflow as tf
import glob
import re
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras import losses, optimizers, regularizers
import h5py

DETECTION_THRESHOLD = 0.5


class TLClassifier(object):
    def __init__(self):
        self.is_site = False
        self.model_type = 'tf'

        # specify path to /models directory with respect to the absolute path of tl_classifier.py
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if self.model_type == 'tf':
            # specify the model name based on the is_site flag state
            if self.is_site:
                file_name = 'ssd_real.pb'
            else:
                file_name = 'ssd_sim.pb'

            # full path to the model file
            frozen_graph_file = os.path.join(model_dir, file_name)

            # Import tensorflow graph
            self.detection_graph = tf.Graph()
            with self.detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(frozen_graph_file, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

                # get all necessary tensors
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')

            self.sess = tf.Session(graph=self.detection_graph)
        elif self.model_type == 'keras':
            if self.is_site:
                file_name = 'keras_real.h5'
            else:
                file_name = 'keras_sim.h5'

            self.model = load_model(os.path.join(model_dir, file_name))

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.model_type == 'tf':
            return self.run_tf_classifier(image)
        elif self.model_type == 'keras':
            return self.run_keras_classifier(image)
        else:
            return TrafficLight.UNKNOWN

    def run_tf_classifier(self, image):
        # Bounding Box Detection.
        tic = time.time()
        with self.detection_graph.as_default():
            # BGR to RGB conversion
            image = image[:, :, ::-1]

            img = Image.fromarray(image.astype('uint8'), 'RGB')
            # size = 640, 480
            # img.thumbnail(size, Image.ANTIALIAS)
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)
            # run classifier
            (scores, classes) = self.sess.run(
                [self.d_scores, self.d_classes],
                feed_dict={self.image_tensor: img_expanded})

            # find the top score for a given image frame
            top_score = np.amax(np.squeeze(scores))

            elapsed_time = time.time() - tic
            sys.stderr.write("Debug: Time spent on classification=%.2f sec\n" % (elapsed_time))

            # figure out traffic light class based on the top score
            if top_score > DETECTION_THRESHOLD:
                tl_state = int(np.squeeze(classes)[0])
                if tl_state == 1:
                    sys.stderr.write("Debug: Traffic state: RED, score=%.2f\n" % (top_score * 100))
                    return TrafficLight.RED
                elif tl_state == 2:
                    sys.stderr.write("Debug: Traffic state: YELLOW, score=%.2f\n" % (top_score * 100))
                    return TrafficLight.YELLOW
                else:
                    sys.stderr.write("Debug: Traffic state: GREEN, score=%.2f\n" % (top_score * 100))
                    return TrafficLight.GREEN
            else:
                sys.stderr.write("Debug: Traffic state: OFF\n")
                return TrafficLight.UNKNOWN

    def run_keras_classifier(self, image):
        image = cv2.resize(image, (64, 32))
        scores = self.model.predict(image)

        top_score = np.max(scores)
        if top_score > DETECTION_THRESHOLD:
            tl_state = np.argmax(scores)
            if tl_state == 0:
                sys.stderr.write("Debug: Traffic state: RED, score=%.2f\n" % (top_score * 100))
                return TrafficLight.RED
            elif tl_state == 1:
                sys.stderr.write("Debug: Traffic state: YELLOW, score=%.2f\n" % (top_score * 100))
                return TrafficLight.YELLOW
            else:
                sys.stderr.write("Debug: Traffic state: GREEN, score=%.2f\n" % (top_score * 100))
                return TrafficLight.GREEN
        else:
            sys.stderr.write("Debug: Traffic state: OFF\n")
            return TrafficLight.UNKNOWN

    def train_keras_classifier(self):
        light_regex = 'sim_(?P<color>[a-z]+)_[0-9]+.jpg'
        X_train = []
        y_train = []
        for light in glob.glob('train/*.jpg'):
            im = cv2.imread(light)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (64, 32))
            match = re.search(light_regex, light)

            X_train.append(im)
            if match.group('color') == 'red':
                y_train.append(0)
            elif match.group('color') == 'yellow':
                y_train.append(1)
            elif match.group('color') == 'green':
                y_train.append(2)
            elif match.group('color') == 'unknown':
                y_train.append(4)

        y_train = to_categorical(y_train)
        X_train = np.array(X_train)

        # Set up model architecture
        num_classes = 3
        model = self.get_simple_model(num_classes)

        loss = losses.categorical_crossentropy
        optimizer = optimizers.Adam()

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=64, epochs=25, verbose=True, validation_split=0.1, shuffle=True)
        score = model.evaluate(X_train, y_train, verbose=0)
        print(score)

        if self.is_site:
            file_name = 'keras_real.h5'
        else:
            file_name = 'keras_sim.h5'

        model.save(file_name)

    @staticmethod
    def get_simple_model(num_classes):
        model = Sequential([
            Conv2D(32, (3, 3), input_shape=(32, 64, 3), padding='same', activation='relu',
                   kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            Flatten(),
            Dense(8, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)),
            Dense(num_classes, activation='softmax')
        ])

        return model

    @staticmethod
    def get_lenet_model(num_classes, keep_prob=0.2):
        model = Sequential([
            Conv2D(8, 5, padding='same', input_shape=(32, 64, 3), activation='relu'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(16, 5, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(32, 5, padding='same', activation='relu'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Flatten(),
            Dense(240, activation='relu'),
            Dropout(keep_prob),
            Dense(168, activation='relu'),
            Dropout(keep_prob),
            Dense(num_classes, activation='softmax'),
        ])

        return model
