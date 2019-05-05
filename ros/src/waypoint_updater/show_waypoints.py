#!/usr/bin/python
"""
Visualize the project

Disclaimer: Original code from https://github.com/Kairos-Automotive/carla-brain
"""

import sys
import math
import threading
from timeit import default_timer as timer
from collections import deque

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPen
from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer

import rospy
from std_msgs.msg import Int32, Bool
from std_msgs.msg import Float32 as Float
from geometry_msgs.msg import PoseStamped
from dbw_mkz_msgs.msg import SteeringCmd, SteeringReport, ThrottleCmd, BrakeCmd
from styx_msgs.msg import TrafficLightArray, Lane, Light
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

DEQUEU_MAX_LEN = 50

matplotlib.use('Qt5Agg')


class Visualization(QtWidgets.QWidget):

    """
    Subscribe to all the ros publisher and show their content
    """

    def __init__(self):
        super(Visualization, self).__init__()
        rospy.init_node('show_waypoints')
        self.lock = threading.Lock()
        self.bridge = CvBridge()
        self.start_time = timer()

        self.final_waypoints = None
        rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb, queue_size=1)

        self.base_waypoints = None
        self.max_x, self.max_y, self.min_x, self.min_y = (0.1, 0.1, 0.0, 0.0)
        rospy.Subscriber('/base_waypoints', Lane, self.base_waypoints_cb, queue_size=1)

        self.steering_cmd = 0
        self.steering_cmd_deq = deque([], maxlen=DEQUEU_MAX_LEN)
        self.steering_cmd_deq_t = deque([], maxlen=DEQUEU_MAX_LEN)
        rospy.Subscriber('/vehicle/steering_cmd', SteeringCmd, self.steering_cmd_cb, queue_size=1)
        self.steering_rep = None
        self.steering_rep_speed_deq = deque([], maxlen=DEQUEU_MAX_LEN*5)
        self.steering_rep_angle_deq = deque([], maxlen=DEQUEU_MAX_LEN*5)
        self.steering_rep_deq_t = deque([], maxlen=DEQUEU_MAX_LEN*5)
        rospy.Subscriber('/vehicle/steering_report', SteeringReport, self.steering_rep_cb, queue_size=1)

        self.throttle_cmd_type = None
        self.throttle_enable = None
        self.throttle_cmd = None
        self.throttle_cmd_deq = deque([], maxlen=DEQUEU_MAX_LEN)
        self.throttle_cmd_deq_t = deque([], maxlen=DEQUEU_MAX_LEN)
        rospy.Subscriber('/vehicle/throttle_cmd', ThrottleCmd, self.throttle_cmd_cb, queue_size=1)
        self.throttle_rep = None
        self.throttle_rep_deq = deque([], maxlen=DEQUEU_MAX_LEN*5)
        self.throttle_rep_deq_t = deque([], maxlen=DEQUEU_MAX_LEN*5)
        rospy.Subscriber('/vehicle/throttle_report', Float, self.throttle_rep_cb, queue_size=1)

        self.brake_cmd_type = None
        self.brake_enable = None
        self.brake_cmd = None
        self.brake_cmd_deq = deque([], maxlen=DEQUEU_MAX_LEN)
        self.brake_cmd_deq_t = deque([], maxlen=DEQUEU_MAX_LEN)
        rospy.Subscriber('/vehicle/brake_cmd', BrakeCmd, self.brake_cmd_cb, queue_size=1)
        self.brake_rep = None
        self.brake_rep_deq = deque([], maxlen=DEQUEU_MAX_LEN*5)
        self.brake_rep_deq_t = deque([], maxlen=DEQUEU_MAX_LEN*5)
        rospy.Subscriber('/vehicle/brake_report', Float, self.brake_rep_cb, queue_size=1)

        self.lights = None
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb, queue_size=1)

        self.traffic_light = - 1
        rospy.Subscriber('/traffic_light', Light, self.traffic_waypoint_cb, queue_size=1)

        self.current_pose = None
        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb, queue_size=1)

        self.dbw_enabled = False
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb, queue_size=1)

        self.img_format_table = {'rgb8': QtGui.QImage.Format_RGB888, 'mono8': QtGui.QImage.Format_Mono,
                                 'bgr8': QtGui.QImage.Format_RGB888}
        self.image = QtGui.QImage(np.zeros([300, 400, 3]), 400, 300, self.img_format_table['bgr8'])
        rospy.Subscriber('/image_color', Image, self.camera_callback, queue_size=1)

        self.initUI()

        self.timer = QTimer()
        self.timer.timeout.connect(self.repaint)
        self.timer.setInterval(300)
        self.timer.start()


    def initUI(self):
        """
        Initialize the gui
        """
        self.setGeometry(10, 10, 1100, 1100)
        self.setWindowTitle('Carla Diagnostics')
        self.initMPL()
        self.show()

    def initMPL(self):
        """
        initialize matplotlib objects
        """
        self.figure = Figure()

        self.throttle_axes = self.figure.add_subplot(411)
        self.throttle_axes.grid(True)

        self.brake_axes = self.figure.add_subplot(412, sharex=self.throttle_axes)
        self.brake_axes.grid(True)

        self.steer_axes = self.figure.add_subplot(413, sharex=self.throttle_axes)
        self.steer_axes.grid(True)

        self.speed_axes = self.figure.add_subplot(414, sharex=self.throttle_axes)
        self.speed_axes.grid(True)

        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout()
        self.plot_widget = QtWidgets.QWidget(self)
        mpl_x = 315
        mpl_y = 395
        mpl_w = 550
        mpl_h = 320
        self.plot_widget.setGeometry(mpl_x, mpl_y, mpl_w, mpl_h)
        self.plot_widget.setLayout(layout)
        layout.addWidget(self.canvas)


    def paintEvent(self, e):
        """
        Paint all the content
        :param e:
        :return:
        """
        painter = QtGui.QPainter()
        painter.begin(self)
        self.drawPoints(painter)
        self.draw_plots()
        painter.end()


    def draw_plots(self):
        """
        re-draw matplotlib plots with fresh data
        """
        self.throttle_axes.clear()
        self.throttle_axes.set_ylim(0., 1.)
        self.throttle_axes.grid(True)
        self.throttle_axes.plot(self.throttle_cmd_deq_t, self.throttle_cmd_deq, 'r', alpha=1.0)
        self.throttle_axes.plot(self.throttle_rep_deq_t, self.throttle_rep_deq, 'b', alpha=0.5)
        self.throttle_axes.set_ylabel("Throttle", fontsize=8)
        self.throttle_axes.set_xlabel("Time", fontsize=8)

        self.brake_axes.clear()
        self.brake_axes.set_ylim(0., 4000.)
        self.brake_axes.grid(True)
        self.brake_axes.plot(self.brake_cmd_deq_t, self.brake_cmd_deq, 'r', alpha=1.0)
        self.brake_axes.plot(self.brake_rep_deq_t, self.brake_rep_deq, 'b', alpha=0.5)
        self.brake_axes.set_ylabel("Brake", fontsize=8)
        self.brake_axes.set_xlabel("Time", fontsize=8)

        self.steer_axes.clear()
        self.steer_axes.grid(True)
        self.steer_axes.plot(self.steering_cmd_deq_t, self.steering_cmd_deq, 'r', alpha=1.0)
        self.steer_axes.plot(self.steering_rep_deq_t, self.steering_rep_angle_deq, 'b', alpha=0.5)
        self.steer_axes.set_ylabel("Steer Angle", fontsize=8)
        self.steer_axes.set_xlabel("Time", fontsize=8)

        self.speed_axes.clear()
        self.speed_axes.set_ylim(0, 15.)
        self.speed_axes.grid(True)
        self.speed_axes.plot(self.steering_rep_deq_t, self.steering_rep_speed_deq, 'b', alpha=0.5)
        self.speed_axes.set_ylabel("Speed", fontsize=8)
        self.speed_axes.set_xlabel("Time", fontsize=8)

        self.canvas.draw()


    def calculate_position(self, orig_x, orig_y):
        """
        Readjust the position to be displayed within window on most screens
        :param orig_x:
        :param orig_y:
        :return:
        """
        mov_x = -1*self.min_x
        mov_y = -1*self.min_y 
        #rospy.logwarn("x and y %r %r   %r %r   %r %r", self.max_x, self.max_y, self.min_x, self.min_y, mov_x, mov_y)

        x = 900 - (orig_x + mov_x) * 900 / (self.max_x - self.min_x) + 150
        y = (orig_y + mov_y) * 900 / (self.max_y - self.min_y) + 40
        return (x, y)

    def draw_traffic_lights(self, painter):
        """
        If traffic lights have been provided, draw them.
        :param painter:
        :return:
        """
        if self.lights:
            pen = QPen()
            pen.setWidth(10)
            pen.setColor(Qt.blue)
            painter.setPen(pen)
            for position in self.lights:
                x, y = self.calculate_position(position.pose.pose.position.x, position.pose.pose.position.y)
                painter.drawPoint(x, y)

    def draw_dbw_enabled(self, painter):
        """
        Are we in manual or automatic mode
        :param painter:
        :return:
        """
        pen = QPen()
        pen.setColor(Qt.black)
        painter.setPen(pen)
        text = "Automatic" if self.dbw_enabled else "Manual"
        painter.drawText(QPointF(10, 20), text)

    def draw_current_pose(self, painter):
        """
        Draw the current position
        :param painter:
        :return:
        """
        if self.current_pose:
            pen = QPen()
            pen.setWidth(15)
            pen.setColor(Qt.darkMagenta)
            painter.setPen(pen)
            x, y = self.calculate_position(self.current_pose.pose.position.x, self.current_pose.pose.position.y)
            painter.drawPoint(x, y)

    def draw_next_traffic_light(self, painter):
        """
        Draw the upcoming traffic light
        :param painter:
        :return:
        """
        twp = self.traffic_light
        if twp >= 0 and self.lights and twp <= len(self.base_waypoints):
            #rospy.logwarn("%r %r", twp, len(self.base_waypoints))
            pen = QPen()
            pen.setWidth(20)
            pen.setColor(Qt.red)
            painter.setPen(pen)
            x, y = self.calculate_position(self.base_waypoints[twp].pose.pose.position.x,
                                           self.base_waypoints[twp].pose.pose.position.y)
            painter.drawPoint(x, y)

    def drawPoints(self, painter):
        """
        Draw the recevied content
        :param painter:
        :return:
        """
        # draw the whole track (base waypoints)
        pen = QPen()
        pen.setWidth(4)
        pen.setColor(Qt.black)
        painter.setPen(pen)
        if self.base_waypoints:
            for waypoint in self.base_waypoints:
                x, y = self.calculate_position(waypoint.pose.pose.position.x,
                                               waypoint.pose.pose.position.y)
                painter.drawPoint(x, y)

        # draw final waypoints published (immediately in front, with the required speed)
        pen = QPen()
        pen.setWidth(6)
        pen.setColor(Qt.red)
        painter.setPen(pen)
        if self.final_waypoints:
            for waypoint in self.final_waypoints:
                x, y = self.calculate_position(waypoint.pose.pose.position.x,
                                               waypoint.pose.pose.position.y)
                painter.drawPoint(x, y)

        # draw steering command and report
        cx = 130
        cy = 130
        r = 100.0
        pen = QPen()
        pen.setWidth(3)
        pen.setColor(Qt.black)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(cx, cy), r, r)
        self.draw_steering(painter, cx, cy, r, 10, self.steering_cmd, Qt.red)
        self.draw_steering_report(painter, cx, cy, r, Qt.blue)
        self.draw_brake_throttle(painter, cx, cy, r, Qt.black)

        # draw debug image with TL regions detected
        if self.image:
            image_x = 450
            image_y = 90
            painter.drawImage(QRectF(image_x, image_y, self.image.size().width(), self.image.size().height()), self.image)

        # draw remaining stats
        self.draw_next_traffic_light(painter)
        self.draw_dbw_enabled(painter)
        self.draw_current_pose(painter)
        self.draw_traffic_lights(painter)


    def draw_steering(self, painter, cx, cy, r, width, steering, color):
        """
        Draw the steering angle
        """
        pen = QPen()
        pen.setWidth(width)
        pen.setColor(color)
        painter.setPen(pen)
        x = cx + r * math.cos(-math.pi / 2 + steering * -1)
        y = cy + r * math.sin(-math.pi / 2 + steering * -1)
        painter.drawLine(QPointF(cx, cy), QPointF(x, y))

    def draw_steering_report(self, painter, cx, cy, r, color):
        """
        Draw the reported steering angle
        """
        if self.steering_rep:
            pen = QPen()
            pen.setColor(Qt.black)
            painter.setPen(pen)
            text = "%4d km/h" % (self.steering_rep.speed * 3.6)
            painter.drawText(QPointF(cx-20, cy+r+20), text)

            self.draw_steering(painter, cx, cy, r, 5, self.steering_rep.steering_wheel_angle, color)

    def draw_brake_throttle(self, painter, cx, cy, r, color):
        """
        Draw brake and throttle commands
        """
        if self.throttle_cmd_type:
            str = '??'
            value = self.throttle_cmd
            if self.throttle_cmd_type == ThrottleCmd.CMD_PERCENT:
                str = '%'
                value *= 100
            elif self.throttle_cmd_type == ThrottleCmd.CMD_PEDAL:
                str = 'pedal'
            pen = QPen()
            pen.setColor(Qt.black)
            painter.setPen(pen)
            text = "Throttle: %2.2f %s " % (value, str)
            painter.drawText(QPointF(cx-20-30, cy+r+20+20), text)

        if self.brake_cmd_type:
            str = '??'
            value = self.brake_cmd
            if self.brake_cmd_type == BrakeCmd.CMD_PEDAL:
                str = 'pedal'
            elif self.brake_cmd_type == BrakeCmd.CMD_PERCENT:
                str = '%'
                value *= 100
            elif self.brake_cmd_type == BrakeCmd.CMD_TORQUE:
                str = 'torque Nm'
            pen = QPen()
            pen.setColor(Qt.black)
            painter.setPen(pen)
            text = "Brake: %5.2f %s " % (value, str)
            painter.drawText(QPointF(cx-20-30, cy+r+20+40), text)


    def final_waypoints_cb(self, msg):
        """
        Callback for /final_waypoints
        :param msg:
        :return:
        """
        self.final_waypoints = msg.waypoints

    def steering_cmd_cb(self, msg):
        """
        Callback for /vehicle/steering_cmd
        :param msg:
        :return:
        """
        self.steering_cmd = msg.steering_wheel_angle_cmd
        # add to deque for plotting
        self.steering_cmd_deq_t.append(timer()-self.start_time)
        self.steering_cmd_deq.append(self.steering_cmd)

    def steering_rep_cb(self, msg):
        """
        Callback for /vehicle/steering_cmd
        :param msg:
        :return:
        """
        self.steering_rep = msg
        # add to deque for plotting
        self.steering_rep_deq_t.append(timer()-self.start_time)
        self.steering_rep_speed_deq.append(self.steering_rep.speed)
        self.steering_rep_angle_deq.append(self.steering_rep.steering_wheel_angle)

    def throttle_cmd_cb(self, msg):
        """
        Callback for /vehicle/throttle_cmd
        :param msg:
        :return:
        """
        self.throttle_enable = msg.enable
        self.throttle_cmd_type = msg.pedal_cmd_type
        self.throttle_cmd = msg.pedal_cmd
        # add to deque for plotting
        self.throttle_cmd_deq_t.append(timer()-self.start_time)
        self.throttle_cmd_deq.append(self.throttle_cmd)

    def throttle_rep_cb(self, msg):
        """
        Callback for /vehicle/throttle_cmd
        :param msg:
        :return:
        """
        self.throttle_rep = msg.data
        # add to deque for plotting
        self.throttle_rep_deq_t.append(timer()-self.start_time)
        self.throttle_rep_deq.append(self.throttle_rep)

    def brake_cmd_cb(self, msg):
        """
        Callback for /vehicle/brake_cmd
        :param msg:
        :return:
        """
        self.brake_enable = msg.enable
        self.brake_cmd_type = msg.pedal_cmd_type
        self.brake_cmd = msg.pedal_cmd
        # add to deque for plotting
        self.brake_cmd_deq_t.append(timer()-self.start_time)
        self.brake_cmd_deq.append(self.brake_cmd)

    def brake_rep_cb(self, msg):
        """
        Callback for /vehicle/brake_cmd
        :param msg:
        :return:
        """
        self.brake_rep = msg.data
        # add to deque for plotting
        self.brake_rep_deq_t.append(timer()-self.start_time)
        self.brake_rep_deq.append(self.brake_rep)

    def base_waypoints_cb(self, msg):
        """
        Callback for /base_waypoints
        :param msg:
        :return:
        """
        self.base_waypoints = msg.waypoints

        max_x = -sys.maxint
        max_y = -sys.maxint
        min_x = sys.maxint
        min_y = sys.maxint
        for waypoint in self.base_waypoints:
            max_x = max(waypoint.pose.pose.position.x, max_x)
            max_y = max(waypoint.pose.pose.position.y, max_y)
            min_x = min(waypoint.pose.pose.position.x, min_x)
            min_y = min(waypoint.pose.pose.position.y, min_y)
        rospy.logwarn("x and y %r %r   %r %r", max_x, max_y, min_x, min_y)
        self.max_x, self.max_y, self.min_x, self.min_y = (max_x, max_y, min_x, min_y)

    def camera_callback(self, data):
        """
        Callback for /image_color
        :param data:
        :return:
        """
        _format = self.img_format_table[data.encoding]
        if data.encoding == "bgr8":
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            image_data = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            image_data = data.data
        image = QtGui.QImage(image_data, data.width, data.height, _format)
        
        self.image = image.scaledToWidth(300)

    def traffic_cb(self, msg):
        """
        Callback for /vehicle/traffic_lights
        :param msg:
        :return:
        """
        self.lights = msg.lights

    def traffic_waypoint_cb(self, msg):
        """
        Callback for /traffic_waypoint
        :param msg:
        :return:
        """
        self.traffic_light = msg.state

    def current_pose_cb(self, msg):
        """
        Callback for /traffic_waypoint
        :param msg:
        :return:
        """
        self.current_pose = msg

    def dbw_enabled_cb(self, msg):
        """
        Callback for /vehicle/dbw_enable
        :param msg:
        :return:
        """
        self.dbw_enabled = msg.data

def main():
    app = QtWidgets.QApplication(sys.argv)
    Visualization()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()