import cv2
import numpy as np
import time
class Robinet:
    def __init__(self, max_flow):
        self.reward = 0
        self.done = False
        self.flow = []
        self.flow_time = []
        self.max_flow = max_flow
        self.start_time = time.time()

    def get_state(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Unable to open camera.")
            return None

        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            return None

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (256, 256))
        cap.release()

        return resized_frame

    def do_step(self, final_action):    # reward, done
        # do thing related to the motor to adjust the flow
        return self.reward, self.done

    def reset(self):
        self.reward = 0
        self.done = False
        self.start_time = time.time()
        self.flow = []

    def set_in_reward(self, satisfaction):
        if satisfaction < 0:
            self.reward = satisfaction * self.flow[-1] / self.max_flow
        else:
            self.reward = - satisfaction / self.flow[-1]

    def set_done_reward(self, satisfaction):
        self.done = True

        somme = 0
        for i in range(len(self.flow) - 1):
            somme += (self.flow_time[i + 1] - self.flow_time[i]) * self.flow[i]
        somme += (self.flow_time[-1] - time.time()) * self.flow[-1]

        self.reward = satisfaction * ((time.time() - self.start_time) * self.max_flow) / (somme * 100)