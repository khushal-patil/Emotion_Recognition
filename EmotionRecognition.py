from tkinter import *

from tkinter import filedialog
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

class EmotionRecognitionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Emotion Recognition")
        self.master.state("zoomed")
        img = Image.open(r"images\bg1.jpg")
        img = img.resize((1280, 180))
        self.photoimg = ImageTk.PhotoImage(img)
        f_banner = Label(self.master, bg="#ffd6cc")
        f_banner.place(x=0, y=0, width=1280, height=50)
        
        title_lbl = Label(self.master, text="EMOTION RECOGNITION SYSTEM", font=(
            "time new roman", 35, "bold"), bg="white", fg="darkblue")
        title_lbl.place(x=0, y=50, width=1280, height=50)

        # background
        bg_img = Label(self.master, bg="#ffd6cc")
        bg_img.place(x=0, y=100, width=1280, height=640)
        
        
        
        tab2 = Image.open(
            r"images\det1.jpg")
        tab2 = tab2.resize((220, 220))
        self.tab_img2 = ImageTk.PhotoImage(tab2)
        b2 = Button(bg_img, command=self.start_live_emotion_recognition,
                    image=self.tab_img2, cursor="hand2")
        b2.place(x=300, y=80, width=300, height=300)
        b2_1 = Button(bg_img, command=self.start_live_emotion_recognition, text="Live Emotion", font=(
            "times new roman", 20), bg="#008000", fg="white")
        b2_1.place(x=300, y=360, width=300, height=50)

        
        tab3 = Image.open(
            r"images\file.png")
        tab3 = tab3.resize((220, 220))
        self.tab_img3 = ImageTk.PhotoImage(tab3)
        b3 = Button(bg_img, command=self.select_image,
                    image=self.tab_img3, cursor="hand2")
        b3.place(x=660, y=80, width=300, height=300)
        b3_1 = Button(bg_img, command=self.select_image, text="Select Image", font=(
            "times new roman", 20), bg="#008000", fg="white")
        b3_1.place(x=660, y=360, width=300, height=50)

        
    def start_live_emotion_recognition(self):
        model = load_model('model.h5')

        video = cv2.VideoCapture(0,cv2.CAP_DSHOW)

        if not video.isOpened():
            print("Error: Unable to open the camera")
            return

        faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

        while True:
            ret, frame = video.read()

            if not ret:
                print("Error: Unable to grab a frame from the video stream")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 3)

            for x, y, w, h in faces:
                sub_face_img = gray[y:y + h, x:x + w]
                resized = cv2.resize(sub_face_img, (48, 48))
                normalize = resized / 255.0
                reshaped = np.reshape(normalize, (1, 48, 48, 1))
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Emotion Recognition", frame)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.run_emotion_recognition(file_path)

    def run_emotion_recognition(self, image_path):
        model = load_model('model.h5')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)
        
        for x, y, w, h in faces:
            sub_face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            print(label)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
       
        cv2.imshow("Emotion Recognition", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
