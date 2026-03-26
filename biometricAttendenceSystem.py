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

known_face_encodings = []
known_face_names = []
for img in files :
    img_path = rf"C:\Users\Harshit\Desktop\Programming\Python Revision\face_recognition\images\{img}"

    image = face_recognition.load_image_file(img_path)

    # Encode Faces
    image_encoding = face_recognition.face_encodings(image)[0]    

    # Store endeing and name 
    known_face_encodings.append(image_encoding)
    known_face_names.append(img.split(".")[0])


# List of expected people 

people = known_face_names.copy()


face_locations = []
face_encodings = []


# get current date and time

now = datetime.datetime.now() 
current_date = now.strftime ("%Y-%m-%d")


f = open(f"{current_date}.csv" , "w+" , newline="")
lnwriter = csv.writer(f)

#Image Optimization

while True:
    ret, frame = video.read()
    if not ret:
        break
    small_frame = cv2.resize(frame , (0,0) , fx = 0.2 , fy = 0.2)
    rgb_small_frame = cv2.cvtColor(small_frame , cv2.COLOR_BGR2RGB)

    #Recognize faces

    face_locations = face_recognition.face_locations(rgb_small_frame , model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame , face_locations)

    for face_encoding in face_encodings :

        matches = face_recognition.compare_faces(known_face_encodings , face_encoding) #This will return list of boolean values 

        face_distance = face_recognition.face_distance(known_face_encodings , face_encoding) # it Identifies how similar the faces are , stores fase distances as list

        best_match_index = np.argmin(face_distance)


        if(matches[best_match_index]):
            name = known_face_names[best_match_index]


        #Add text if the person is present 

        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10 , 100)
            fontScale = 1.5
            fontColor = (255 , 0 , 0)
            thickness = 3
            lineType = 2
            cv2.putText (frame , name + "Present" , bottomLeftCornerOfText , font , fontScale , fontColor , thickness , lineType )

            if name in people :
                people.remove(name) 
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name , current_time])


    cv2.imshow("Attendance" , frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release() 
cv2.destroyAllWindows()  
f.close()