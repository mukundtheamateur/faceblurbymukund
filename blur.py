import cv2
import sys
import numpy as np

#creating the required instances
cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

#specifies the type of blur user wants
blur_type = ""

#handing whether the user have given CLA
try:
    x = sys.argv
    blur_type = x[1]
except:
    print("You have not specified the type of blur you want !")
    exit(-1)

#this function does gaussian blur
def blur_gaussian(image,factor=3.0):
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
	
    if kW % 2==0:
        kW -= 1
    if kH % 2==0:
	    kH -= 1
    return cv2.GaussianBlur(image, (kW, kH), 0)

#this function does pixelate blur
def blur_pixelate(image,blocks=10):
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),(B, G, R), -1)
    return image

#this function detects faces
def detect_faces(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    return faces

#this is the main function
def main():

    if blur_type!="gaussian" and blur_type!="pixelate":
        print("Possible error with the command Line Argument !!!")
        exit(-1)

    while True:
        #capturing frame
        ret, frame = cam.read()

        if ret is True:
            faces = detect_faces(frame)

            for face in faces:
                (x,y,w,h) = face
                if blur_type=="gaussian":
                    frame[y:y+h, x:x+w] = blur_gaussian(frame[y:y+h, x:x+w])
                elif blur_type=="pixelate":
                    frame[y:y+h, x:x+w] = blur_pixelate(frame[y:y+h, x:x+w])

            cv2.imshow("Face blur by Mukund",frame)
        else:
            print("Problem with your webcam !")
            exit(-1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
    cam.release()
    cv2.destroyAllWindows()