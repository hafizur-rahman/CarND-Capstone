#!/usr/bin/env python
import sys
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import time
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

LOGGING_THROTTLE_FACTOR = 5  # Only log at this rate (1 / Hz)

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.camera_image = None
        self.lights = []

        self.process_count = 0
        self.has_image = False

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.last_classify_time = time.time()
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        ## Image capturing
        self.train_img_idx = 0
        self.sim_image_grab_max_range = 80     # Only grab image when close to traffic light
        self.sim_image_grab_min_range = 2      # But not too close
        self.sim_image_grab_min_spacing = 1    # Distance gap between images
        self.image_grab_last_light = None      # Identify which light we were approaching last time
        self.image_grab_last_distance = 0      # Distance from light last time

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

        if time.time() - self.last_classify_time < 0.2:
            return

        light_wp, state = self.process_traffic_lights()
        self.last_classify_time = time.time()

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
            light_wp = light_wp if (state == TrafficLight.RED or state == TrafficLight.YELLOW) else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
            # Note: changed to x,y coords as suggested by walkthrough video.

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        if self.waypoint_tree:
            return self.waypoint_tree.query([x, y], 1)[1]
        else:
            return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light_state_sim = light.state

        if (not self.has_image):
            self.prev_light_loc = None
        else:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

            save_image = False
            #save_image = self.good_position_to_save_sim_image(light)
            if save_image:
                # Grab images for DL model training
                file_name = "/capstone/data/train/sim_{}_{}.jpg".format(
                    self.to_string(light_state_sim), self.train_img_idx)
                
                cv2.imwrite(file_name, cv_image)
                sys.stderr.write("Debug tl_detector: saved training image " + file_name + "\n")

                self.train_img_idx += 1 
            
        # Get classification
        return self.light_classifier.get_classification(cv_image)

        # For testing, just return the light state
        #return light_state_sim

    def good_position_to_save_sim_image(self, closest_light):
        """Considers whether we are within the range limits before a traffic light
           to save a training image, and also whether we have moved far enough since
           the last one to save a new image, so that we only end up saving a reasonable
           number of training images from the simulation and only ones that have
           traffic lights in.
           
        Returns:
           bool: True if now is a good time to save a training image"""
          
       
        # Figure out 2D Euclidean distance between us and this closest light
        delta_x = self.pose.pose.position.x - closest_light.pose.pose.position.x
        delta_y = self.pose.pose.position.y - closest_light.pose.pose.position.y
        dist_sqd = delta_x * delta_x + delta_y * delta_y
        distance = math.sqrt(dist_sqd)
        

        if (self.sim_image_grab_min_range <= distance <= self.sim_image_grab_max_range):
            # We're within a suitable range of the light we're approaching
            if (self.image_grab_last_light is None or
                self.image_grab_last_light.state != closest_light.state):
                # Definitely grab image if first light we've found, or it has changed colour
                do_grab_image = True
            elif closest_light.pose.pose.position.x != self.image_grab_last_light.pose.pose.position.x:
                # First time we've been in range for this particular light so
                # we definitely want to grab it (bit lazy to use exact equality of
                # coordinate but works OK; header.seq always zero so no use)
                #sys.stderr.write("dist=%f first time this light True\n" % distance)
                do_grab_image = True
            elif distance <= self.image_grab_last_distance - self.sim_image_grab_min_spacing:
                # We have approached the light more closely than the last time we
                # grabbed an image by enough distance for it to be worth capturing a new image
                #sys.stderr.write("dist=%f got closer so True\n" % distance)
                do_grab_image = True
            else:
                # We have not moved enough since last time, so skip this time as the
                # image will be more or less the same as the last time
                #sys.stderr.write("dist=%f not much closer so False\n" % distance)
                do_grab_image = False
        else:
            # We're not in the right distance bracket but make sure we get first
            # image when we do get within range
            self.image_grab_last_light_x = 0
            #sys.stderr.write("Debug: dist=%f outside limits so False\n" % distance)
            do_grab_image = False

        if do_grab_image:
            self.image_grab_last_light = closest_light
            self.image_grab_last_distance = distance
            
        return do_grab_image

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = -1
        state = TrafficLight.UNKNOWN        

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

        # find the closest visible traffic light (if one exists)
        diff = len(self.waypoints.waypoints)
        for i, light in enumerate(self.lights):
            # Get stop line waypoint index
            line = stop_line_positions[i]
            temp_wp_idx = self.get_closest_waypoint(line[0], line[1])

            # find closest stop line waypoint index
            d = temp_wp_idx - car_position
            if d >=0 and d < diff:
                diff = d
                closest_light = light
                line_wp_idx = temp_wp_idx

        if closest_light:
            self.process_count += 1
            state = self.get_light_state(closest_light)

        return line_wp_idx, state


    def to_string(self, state):
        out = "unknown"
        if state == TrafficLight.GREEN:
            out = "green"
        elif state == TrafficLight.YELLOW:
            out = "yellow"
        elif state == TrafficLight.RED:
            out = "red"
        return out

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
