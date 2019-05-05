import rospy
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, max_speed, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel,
                max_steer_angle):


        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1,
                                            max_lat_accel, max_steer_angle)

        kp = 0.4
        ki = 0.1
        kd = 0.
        mn = 0.  # Minimum throttle value
        mx = max_speed * 0.005 # Maximum throttle value
        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 0.02 # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass + fuel_capacity*GAS_DENSITY
        self.fuel_capaciy = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()   

    def control(self, target_velocity, current_velocity):
        current_vel = self.vel_lpf.filt(current_velocity.linear.x)
        
        steering = self.yaw_controller.get_steering(
            target_velocity.linear.x,
            target_velocity.angular.z,
            current_velocity.linear.x)

        vel_error = target_velocity.linear.x - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0.0

        if target_velocity.linear.x == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400
        elif vel_error < 0.:
            decel = max(vel_error, self.decel_limit)
            brake = 0.2*abs(decel)*self.vehicle_mass*self.wheel_radius # Torque N*m
           
        if vel_error == 0.0 and throttle == 0.0:
            brake = 0.0

        return throttle, brake, steering

    def reset(self):
        self.throttle_controller.reset()        