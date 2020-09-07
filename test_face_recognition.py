from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import time
from moviepy.editor import VideoFileClip
import datetime

def calculateAvgEmotion(emotions):
	total_frames=sum(emotions.values())
	for key,value in emotions.items():
		percent=value*100.0/total_frames
		ans="\n{}:\tPercentage:{}%\n"
		print(ans.format(key,round(percent,2)))

def write_to_folder(c,f,l):

	path="FRAMES_category_Emotions/"+l+"/frame%d.jpg"%c
	# print(path)
	cv2.imwrite(path,frame)
	            

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier =load_model('Emotion.h5')

class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

video_name="FRAMES_category_Emotions/EmotionVideo.mp4"
cap = cv2.VideoCapture(video_name)
clip=VideoFileClip(video_name)
duration=clip.duration


###stop duration
stop_at=int(duration-20)

# use the following line after commenting the 
# above line for using with webcam
# cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('FRAMES_category_Emotions/outfile_emotions.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))


emotions={"Angry":0,"Disgust":0,"Fear":0,"Happy":0,"Neutral":0,"Sad":0,"Surprise":0}
start=time.time()


count=0
face_count=0

while True:
    ret, frame = cap.read()
    labels = []
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if ret:
	    for (x,y,w,h) in faces:
	        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	        roi_gray = gray[y:y+h,x:x+w]
	        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

	        if np.sum([roi_gray])!=0:
	            roi = roi_gray.astype('float')/255.0
	            roi = img_to_array(roi)
	            roi = np.expand_dims(roi,axis=0)

	            preds = classifier.predict(roi)[0]
	            label=class_labels[preds.argmax()]
	            label_position = (x,y)
	            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
	            emotions[label]+=1
	            out.write(frame)

	            write_to_folder(count,frame,label)	         
	            count+=1
				
	        else:
	            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
	            out.write(frame)
	            count+=1
				


	    cv2.imshow('Emotion Detector',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	    print(round(time.time()-start))
	    if time.time()-start > stop_at:
	    	break
    else:
    	break

calculateAvgEmotion(emotions)

cap.release()
out.release()
cv2.destroyAllWindows()

























