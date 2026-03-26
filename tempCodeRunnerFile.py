import face_recognition
import cv2
import numpy as np
import csv 
import datetime
import os



# Loads Camers
 
video = cv2.VideoCapture(0)

#Load Known Images
files = os.listdir(r"C:\Users\Harshit\Desktop\Programming\Python Revision\face_recognition\images")
print(files)