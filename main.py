import cv2
import numpy as np
import face_recognition

img = face_recognition.load_image_file('Images/Gunjan.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images/Gunjan Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(img)[0]
encode = face_recognition.face_encodings(img)[0]
cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encode], encodeTest)

faceDis = face_recognition.face_distance([encode], encodeTest)
print(results)

cv2.putText(imgTest, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

cv2.imshow('Gunjan', img)
cv2.imshow('Gunjan Test', imgTest)
cv2.waitKey(0)