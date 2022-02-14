import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

model=tf.keras.models.load_model('E:/PY/face_emotion/emotions_fromvid/face_emotion_high_valacc_input_48x48x1.h5')

path='E:/PY/face_emotion/emotions_fromvid/expressions2.mp4'
cap = cv2.VideoCapture(path)  #set path to zero for live webcam
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Output_expressions2_sad.avi',fourcc, 10.0, (1000,720))

face_cascade = cv2.CascadeClassifier('E:/PY/face_emotion/emotions_fromvid/haarcascade_frontalface_default.xml')

def get_emotion(cropped_arr):
    emo=model.predict_classes(np.array([cropped_arr]))[0]
    emotions=['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    return emotions[int(emo)]

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if (ret==True):
        if (i>356 and i<387) or (i>557 and i<575) or(i>650 and i<690) or (i>742 and i<771):  # i = frame number eg. from frames 1 to 20, 30 to 50 etc.
            #frame=cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)        
            #frame = cv2.flip(frame,0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.23,minNeighbors=8)#,minSize=(50,50))
            for (x,y,w,h) in faces:
                frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cropped_gray = gray[y:y+h, x:x+w]
                cropped_color = frame[y:y+h, x:x+w]
                cropped=cv2.resize(cropped_gray,(48,48))
                cropped_arr=np.reshape(cropped,(48,48,1))/255.
                emotion=get_emotion(cropped_arr)
                font = cv2.FONT_HERSHEY_SIMPLEX 
                color = (0,255, 0)
                org = (x+w, y+h)
                fontScale = 1
                thickness=2
                frame=cv2.putText(frame, emotion, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA)

            frame=cv2.resize(frame,(1000,720))
            out.write(frame)
            cv2.imshow('frame',frame)
            i+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            i+=1
    else:
        break

# Release everything if job is finished
print(i)
cap.release()
out.release()
cv2.destroyAllWindows()