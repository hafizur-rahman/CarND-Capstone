#!/usr/bin/env python
import sys
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane, Light
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import time

import numpy as np
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3
LOOKAHEAD_WPS = 150

TRAFFIC_LIGHT_STATES = [
    "RED", "YELLOW", "GREEN", "UNKNOWN", "UNKNOWN"
]

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.current_position = None
        self.current_car_index = None

        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None        

        self.all_traffic_lights = []
        self.all_traffic_light_indices = []
        self.stop_line_indices = []

        self.process_count = 0
        self.has_image = False

        self.bridge = CvBridge()

        model_path = rospy.get_param('~model_path')
        self.light_classifier = TLClassifier(model_path)
        self.listener = tf.TransformListener()
        
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0 

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.stop_line_positions = self.config['stop_line_positions']

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_light_forward', Light, queue_size=1)

        rospy.spin()

    def pose_cb(self, msg):
        self.current_position = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
        self.waypoint_tree = KDTree(self.waypoints_2d)


    def traffic_cb(self, msg):
        self.all_traffic_lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_light_forward

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        
        light_wp, state = self.process_traffic_lights()
        light = Light()
 
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            
            if state != TrafficLight.RED and state != TrafficLight.YELLOW:
                light_wp = -1

            self.last_wp = light_wp	    
            light.index = light_wp
            light.state = state 
            self.upcoming_red_light_pub.publish(light)
        else:
            light.index = self.last_wp
            light.state = self.last_state 
            self.upcoming_red_light_pub.publish(light)
            self.state_count += 1

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """        
        if not (self.waypoints_2d and self.waypoint_tree and self.current_position):
            return -1, TrafficLight.UNKNOWN

        # Initialize indices
        if len(self.all_traffic_light_indices) == 0 and self.base_waypoints:
            self.update_traffic_light_indices()
            self.update_stop_line_indices()

        # Find current car index
        self.current_car_index = self.get_nearest_waypoint_index(
            self.current_position.pose.position.x, self.current_position.pose.position.y)
        
        # Find nearest traffic light info
        tl_index, stop_line_index = self.get_nearest_traffic_light()        
        if not tl_index or (tl_index - self.current_car_index) > LOOKAHEAD_WPS:
            rospy.loginfo('No traffic light in sight')

            return -1, TrafficLight.UNKNOWN

        state = self.get_light_state()
        rospy.loginfo('Traffic light in state : {}'.format(TRAFFIC_LIGHT_STATES[state]))
        
        return stop_line_index, state

    def get_nearest_traffic_light(self):
        for i in range(len(self.all_traffic_light_indices)):
            traffic_light_index = self.all_traffic_light_indices[i]

            if traffic_light_index > self.current_car_index:
                rospy.loginfo('(Current Car Index, Nearest Stop line index): ({}, {})'.format(
                    self.current_car_index, self.stop_line_indices[i]))
                
                return traffic_light_index, self.stop_line_indices[i]
        return None, None

    def update_traffic_light_indices(self):
        if self.all_traffic_lights:
            for light in self.all_traffic_lights:
                nearest_index = self.get_nearest_waypoint_index(light.pose.pose.position.x, 
                                                                light.pose.pose.position.y)
                self.all_traffic_light_indices.append(nearest_index)

    def update_stop_line_indices(self):
        for line in self.stop_line_positions:            
            nearest_index = self.get_nearest_waypoint_index(line[0], line[1])
            self.stop_line_indices.append(nearest_index)

    def get_nearest_waypoint_index(self, x, y):
        '''Returns the nearest(ahead of the current position) waypoint index from the current pose.'''
        nearest_index = self.waypoint_tree.query([x, y], 1)[1]

        #check if closest is ahead or behind vehicle
        nearest_coordinate = self.waypoints_2d[nearest_index]
        prev_coordinate = self.waypoints_2d[nearest_index-1]

        # equation for hyperplane through closest coords
        cl_vect = np.array(nearest_coordinate)
        prev_vect = np.array(prev_coordinate)
        pos_vect = np.array([x, y])

        val = np.dot((cl_vect - prev_vect),(pos_vect - cl_vect))

        if val > 0:
            nearest_index = (nearest_index + 1) % len(self.waypoints_2d)

        return nearest_index

    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (self.has_image):
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            # Get classification
            return self.light_classifier.get_classification(cv_image)
        else:
            return self.last_state

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
