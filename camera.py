import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from pandastable import Table, TableModel
import datetime
from threading import Thread

# from Spotipy import *
import time
import pandas as pd

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor = 0.6

emotion_model = load_model("model.h5")

cv2.ocl.setUseOpenCL(False)

emotion_dict = {
    0: "Happy",
    1: "Sad",
}
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]


""" Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming """


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


""" Class for using another thread for video streaming to boost performance """


class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


""" Class for reading video stream, generating prediction and recommendations """


class VideoCamera(object):
    def get_frame(self):
        global cap1
        global df1
        cap1 = WebcamVideoStream(src=0).start()
        image = cap1.read()
        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
            roi_gray_frame = gray[y : y + h, x : x + w]
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0
            )
            prediction = emotion_model.predict(cropped_img)

            max_index = np.argmax(predictions[0])
            predicted_emotion = emotion_dict[max_index]
            print(max_index, prediction, predicted_emotion)
            cv2.putText(
                image,
                predicted_emotion,
                (x + 20, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if emotion_word == "sad":
                emotion_code = 0
            else:
                emotion_code = 1
            df = get_results(emotion_code)

        global last_frame1
        last_frame1 = image.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(last_frame1)
        img = np.array(img)
        ret, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes(), df


def get_results(emotion_code):
    df = pd.read_csv('songs\culster_data.csv')
    NUM_RECOMMEND = 15
    if emotion_code == 0:
        happy_set = (df[df["kmeans"] == 0][["name", "artists", "year"]].head(NUM_RECOMMEND))
        return pd.DataFrame(happy_set)
    else:
        sad_set = (df[df["kmeans"] == 1][["name", "artists", "year"]].head(NUM_RECOMMEND))
        return pd.DataFrame(sad_set)
