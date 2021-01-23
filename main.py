import os
import cv2
import numpy as np
import face_recognition
import time
from IPython.display import clear_output

path = "images"
images = []
classNames = []

myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])


def findEncodingsImages(images):
    encodeList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodingsImages(images)


def recogniseMe():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            y1, x1, y2, x2 = [element * 4 for element in faceLoc]
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x2, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
                print('welcome back Mr. ' + name, end='\n')
                cv2.destroyAllWindows()
                time.sleep(3)
                clear_output(wait=True)
                return True
            else:
                name = 'unknown'
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, name, (x2, y2 - 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
                print('please try put your face in the camera', end='\n')

        cv2.imshow('webcam', img)
        cv2.waitKey(1)


codes = ['0000', '1111', '2222']
code = ""

while True:
    code = input('enter your code : \n')
    while code == "":
        code = input('enter your code : \n')
    else:
        while code not in codes:
            code = input('wrong code please enter your code again : \n')
        else:
            code = ""
            print('please show your face to be sure', end='\n')
            recogniseMe()
            clear_output(wait=True)
