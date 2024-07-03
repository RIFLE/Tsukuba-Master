import numpy as np
from scipy.interpolate import interp1d

import RPi.GPIO as GPIO
import time

class TouchScreen:
    def __init__(self):
        
        # SETUP
        #####
        self.CANVAS_DIMS = [500,500]

        self.MODE = 0 # 0 = calibration, 1 = drawing
        self.POSITION_MODE = 1 # 1 = screen, 0 = mouse

        self.sensorL = [0,0]
        self.sensorR = [0,45]

        self.BOUNDS = [[0,0], self.CANVAS_DIMS]
        self.CALIBRATION_POINT1 = [50,50]
        self.CALIBRATION_POINT2 = [self.CANVAS_DIMS[1]-50,self.CANVAS_DIMS[0]-50]

        self.XY_1 = [0, 10] # Upper left corner in cm
        self.XY_2 = [45, 80] # Lower right corner in cm

        self.CALIBRATED_1 = False
        self.CALIBRATED_2 = False

        interp_x = interp1d([self.XY_1[0],self.XY_2[0]],[self.CALIBRATION_POINT1[0],self.CALIBRATION_POINT2[0]], bounds_error=False, fill_value="extrapolate")
        interp_y = interp1d([self.XY_1[1],self.XY_2[1]],[self.CALIBRATION_POINT1[1],self.CALIBRATION_POINT2[1]], bounds_error=False, fill_value="extrapolate")
        self.map_xy = [interp_x, interp_y]
        ######

        ## SETTING UP GPIO
        #set GPIO Pins
        if self.POSITION_MODE == 0:
            self.GPIO_TRIGGER_L = 18
            self.GPIO_ECHO_L = 24

            self.GPIO_TRIGGER_R = 20
            self.GPIO_ECHO_R = 21
        elif self.POSITION_MODE == 1:
            self.GPIO_TRIGGER_R = 18
            self.GPIO_ECHO_R = 24

            self.GPIO_TRIGGER_L = 20
            self.GPIO_ECHO_L = 21
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.GPIO_TRIGGER_R, GPIO.OUT)
        GPIO.setup(self.GPIO_ECHO_R, GPIO.IN)

        GPIO.setup(self.GPIO_TRIGGER_L, GPIO.OUT)
        GPIO.setup(self.GPIO_ECHO_L, GPIO.IN)
        ######
    
    def cleanup(self):
        GPIO.cleanup()

    def xy_to_pixel(self, xy, ignore_bounds=False):
        try:
            i = self.map_xy[0](xy[0]) 
            j = self.map_xy[1](xy[1]) 
            if ignore_bounds:
                return int(i), int(j)
            if i>=self.BOUNDS[0][0] and i<=self.BOUNDS[1][0] and j>=self.BOUNDS[0][1] and j<=self.BOUNDS[1][1]:
                return int(i),int(j)
            else:
                return -1
        except:
            return -1


    def trilaterate(self, d1, d2):
        x = (d1**2 - d2**2)/(2*self.sensorR[1]) + self.sensorR[1]/2
        y = (d1**2 - x**2)**(1/2)
        return [x,y]
     
    def distance(self, TRIGGER, ECHO):
        # set Trigger to HIGH
        GPIO.output(TRIGGER, True)
     
        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        GPIO.output(TRIGGER, False)
     
        StartTime = time.time()
        StopTime = time.time()
     
        # save StartTime
        while GPIO.input(ECHO) == 0:
            StartTime = time.time()
     
        # save time of arrival
        while GPIO.input(ECHO) == 1:
            StopTime = time.time()
     
        # time difference between start and arrival
        TimeElapsed = StopTime - StartTime
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        distance = (TimeElapsed * 34300) / 2
     
        return distance

    def get_dists(self):
            dist_R = self.distance(self.GPIO_TRIGGER_R, self.GPIO_ECHO_R)
            time.sleep(0.001)
            dist_L = self.distance(self.GPIO_TRIGGER_L, self.GPIO_ECHO_L)
            
            return dist_L, dist_R
    
    def calibrate_point(self, screen_width, screen_height, point): # Point can be either 'tl' or 'br'
        CALIBRATED = False
        self.CANVAS_DIMS = [screen_width, screen_height]
        self.CALIBRATION_POINT2 = [self.CANVAS_DIMS[0]-50,self.CANVAS_DIMS[1]-50]
        self.BOUNDS = [[0,0], self.CANVAS_DIMS]
        
        dists_L = [0, 0, 0, 0, 0, 0]
        dists_R = [0, 0, 0, 0, 0, 0]
        N = 0

        if point == "tl":
            print("Choose point (0,0)")
        elif point == "br":
            print(f"Choose point {self.CANVAS_DIMS}")
    
        position_recorder = [-1,-1,-1,-1,-1,-1]
        while True:
            # Obtain distances from left and right sensors
            dist_L, dist_R = self.get_dists()

            # Smoothing noise with moving average
            dists_L[N%len(dists_L)] = dist_L
            dists_R[N%len(dists_R)] = dist_R
            dist_L = (dist_L+sum(dists_L))/(len(dists_L)+1)
            dist_R = (dist_R+sum(dists_R))/(len(dists_R)+1)

            xy = self.trilaterate(dist_L, dist_R)
            print(f"Current position:{xy}")


            pixels = self.xy_to_pixel(xy, ignore_bounds=True) # Converting "real world" xy coordinates to pixel coordinates on the screen
            position_recorder[N%len(position_recorder)] = pixels
            if -1 not in position_recorder:
                # Check if user is holding in this point for time enough
                position_std = np.std(position_recorder, axis=0)
                if position_std[0] < 2 and position_std[1] < 2:
                    if point == "tl":
                        self.XY_1 = xy
                        print("First reference point calibrated")

                    elif point == "br":
                        self.XY_2 = xy
                        print("Second reference point calibrated")
                    
                    return
            N += 1

            time.sleep(0.1) # Interval between measures

    def recalibrate(self):
            interp_x = interp1d([self.XY_1[0],self.XY_2[0]],[self.CALIBRATION_POINT1[0],self.CALIBRATION_POINT2[0]], bounds_error=False, fill_value="extrapolate")
            interp_y = interp1d([self.XY_1[1],self.XY_2[1]],[self.CALIBRATION_POINT1[1],self.CALIBRATION_POINT2[1]], bounds_error=False, fill_value="extrapolate")
            self.map_xy = [interp_x, interp_y]


    def calibrate(self, screen_width, screen_height):
        CALIBRATED_1 = False
        CALIBRATED_2 = False
        self.CANVAS_DIMS = [screen_width, screen_height]
        
        dists_L = [0, 0, 0, 0, 0, 0]
        dists_R = [0, 0, 0, 0, 0, 0]
        N = 0

        print("Choose point (0,0)")

        position_recorder = [-1,-1,-1,-1,-1,-1]
        while True:
            # Obtain distances from left and right sensors
            dist_L, dist_R = self.get_dists()

            # Smoothing noise with moving average
            dists_L[N%len(dists_L)] = dist_L
            dists_R[N%len(dists_R)] = dist_R
            dist_L = (dist_L+sum(dists_L))/(len(dists_L)+1)
            dist_R = (dist_R+sum(dists_R))/(len(dists_R)+1)

            xy = self.trilaterate(dist_L, dist_R)
            print(f"Current position:{xy}")


            pixels = self.xy_to_pixel(xy) # Converting "real world" xy coordinates to pixel coordinates on the screen
            position_recorder[N%len(position_recorder)] = pixels
            if -1 not in position_recorder:
                # Check if user is holding in this point for time enough
                position_std = np.std(position_recorder, axis=0)
                if position_std[0] < 2 and position_std[1] < 2:
                    if not CALIBRATED_1:
                        self.XY_1 = xy
                        CALIBRATED_1 = True
                        position_recorder = [-1,-1,-1,-1,-1,-1]
                        
                        print("First reference point calibrated")
                        print(f"Choose point {self.CANVAS_DIMS}")

                    elif not CALIBRATED_2:
                        self.XY_2 = xy
                        CALIBRATED_2 = True
                        position_recorder = [-1,-1,-1,-1,-1,-1]
                        
                        interp_x = interp1d([self.XY_1[0],self.XY_2[0]],[self.CALIBRATION_POINT1[0],self.CALIBRATION_POINT2[0]], bounds_error=False, fill_value="extrapolate")
                        interp_y = interp1d([self.XY_1[1],self.XY_2[1]],[self.CALIBRATION_POINT1[1],self.CALIBRATION_POINT2[1]], bounds_error=False, fill_value="extrapolate")
                        self.map_xy = [interp_x, interp_y]
                        
                        # Setting to drawing mode
                        print("Calibration complete!")
                        return
            N += 1

            time.sleep(0.1) # Interval between measures

    def position(self):
        dist_L, dist_R = self.get_dists()
        xy = self.trilaterate(dist_L, dist_R)
        return xy
    
    def pixels(self):
        dist_L, dist_R = self.get_dists()
        xy = self.trilaterate(dist_L, dist_R)
        return self.xy_to_pixel(xy)




if __name__ == '__main__':
    try:
        t = TouchScreen()
        t.calibrate()
        while True:
            xy = t.position()
            print(xy, t.xy_to_pixel(xy))
            time.sleep(1)
 
        # Reset by pressing CTRL + C
    except KeyboardInterrupt:
        print("Measurement stopped by User")

