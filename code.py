import mediapipe as mp
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import imutils
chr=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I',
 'J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
model=load_model('text_rec.h5')
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def get_letters(img):
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1,32,32,1)
        ypred = model.predict(thresh)

        x = np.argmax(ypred)
        letters.append(chr[x])
    return letters, image

def get_word(letter):
    word = "".join(letter)
    return word


ml = 150
max_x, max_y = 490, 65
curr_tool = "-"
word="-"
time_init = True
rad = 40
var_inits = False
thick = 10
prevx, prevy = 0,0

#get tools function
def getTool(x):
	if x > 150 and x<250:
		return "erase"

	elif x>270 and x<370:
		return "draw"

	elif x >390 and x <490:
		return"predict"


def index_raised(yi, y9):
	if (y9 - yi) > 40:
		return True

	return False



hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils


mask = np.ones((480, 640))*255
mask = mask.astype('uint8')


cap = cv2.VideoCapture(0)
while True:
	_, frm = cap.read()
	frm = cv2.flip(frm, 1)

	rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

	op = hand_landmark.process(rgb)

	if op.multi_hand_landmarks:
		for i in op.multi_hand_landmarks:
			draw.draw_landmarks(frm, i, hands.HAND_CONNECTIONS)
			x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)
			#print(x,y)

			if x < max_x and y < max_y and x > ml:
				if time_init:
					ctime = time.time()
					time_init = False
				ptime = time.time()

				cv2.circle(frm, (x, y), rad, (0,255,255), 2)
				rad -= 1

				if (ptime - ctime) > 0.8:
					curr_tool = getTool(x)
					print("Present Tool: ", curr_tool)
					time_init = True
					rad = 40

			else:
				time_init = True
				rad = 40

			if curr_tool == "draw":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
					prevx, prevy = x, y

				else:
					prevx = x
					prevy = y




			elif curr_tool == "predict":
				cv2.imwrite("mask.jpg",mask)
				letter,image = get_letters("./mask.jpg")
				word = get_word(letter)
				print(word)
				n=open("./text.txt","a+")
				n.write("\n")
				n.write(str(word))
				plt.imshow(image)
				for i in range(60):
					dd = cap.read()

			elif curr_tool == "erase":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.circle(frm, (x, y), 30, (0,0,0), -1)
					cv2.circle(mask, (x, y), 30, 255, -1)



	op = cv2.bitwise_and(frm, frm, mask=mask)
	frm[:, :, 1] = op[:, :, 1]
	frm[:, :, 2] = op[:, :, 2]


	cv2.putText(frm, "Mode: "+curr_tool, (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
	cv2.putText(frm, "Prediction: "+word, (330,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

	frm = cv2.rectangle(frm, (145,0), (495,70), (200,0,0), -1)
	frm = cv2.rectangle(frm, (150,10), (250,65),(0,50,0) , -1)   
	frm = cv2.rectangle(frm, (270,10), (370,65), (0,0,150), -1)
	frm = cv2.rectangle(frm, (390,10), (490,65), (0,200,0), -1)

	cv2.putText(frm, "ERASE", (170, 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frm, "DRAW", (300, 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

	cv2.putText(frm, "PREDICT", (400, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


	cv2.imshow("paint app", frm)
	cv2.imshow("msk",mask)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break
 