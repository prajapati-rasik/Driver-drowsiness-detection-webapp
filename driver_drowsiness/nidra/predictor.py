#from keras.models import load_model
import imutils
import requests
import cv2
import numpy as np
from django.conf import settings
import numpy as np
import threading
import queue

class Predictor(object):

    # if score goes greater than 15 then ring the alarm
    download_thread = None
    default_status = 'Detecting...'
    myQueue = queue.Queue()

    leye = cv2.CascadeClassifier("./nidra/haar_cascade/haarcascade_lefteye_2splits.xml")
    reye = cv2.CascadeClassifier("./nidra/haar_cascade/haarcascade_righteye_2splits.xml")

    API = 'https://driver-drowsiness-detection-api.onrender.com'

    def storeInQueue(f):
        def wrapper(*args):
            Predictor.myQueue.put(f(*args))
        return wrapper

    @staticmethod
    def predict(frame):
        height, width = frame.shape[:2]

        # converting frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # now we detect face , left_eye and right_eye , see dataset.ipynb for more 
        left_eye = Predictor.leye.detectMultiScale(gray)
        right_eye = Predictor.reye.detectMultiScale(gray)

        if Predictor.download_thread is not None:
            if Predictor.download_thread.is_alive():
                return Predictor.default_status
            
        Predictor.download_thread = threading.Thread(
            target=Predictor.function_that_do, 
            name="worker", 
            args=[left_eye, right_eye, frame]
        )
        Predictor.download_thread.start()
        
        if Predictor.myQueue.qsize() == 0:
            return Predictor.default_status
        
        response = Predictor.myQueue.get()
        if response:
            return 'Closed'
        else:
            return 'Open'

    @staticmethod
    @storeInQueue
    def function_that_do(left_eye, right_eye, frame):
        l_eye = np.zeros((24, 24))
        r_eye = np.zeros((24, 24))
       
        for (x, y, w, h) in right_eye:
            r_eye = frame[y:y + h, x:x + w]  # extracting right eye from frame
            # converting to gray scale
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            # resizing image to (24,24) which input size for our model
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)
            r_eye = r_eye.reshape(1, 24, 24, 1)
            break

        for (x, y, w, h) in left_eye:
            l_eye = frame[y:y + h, x:x + w]  # rxtracting left eye from frame
            # converting to gray scale
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.resize(l_eye, (24, 24))  # resizing image
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)
            l_eye = l_eye.reshape(1, 24, 24, 1)
            break

        reqq = {
            'left_eye': l_eye.tolist(),
            'right_eye': r_eye.tolist()
        }

        url = Predictor.API
        response = requests.post(url, json=reqq)
        response.text.strip()
        flag = response.text[12:16] == "true"
        return flag
