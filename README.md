# faceblurbymukund

yo can do this project without creating an environment but it is good practice , so i did this project in an environment:

--> IN THIS PROJECT WE WILL USE ANACONDA FOR CREATING A VIRTUAL ENVIRONMENT.

install anaconda from link : "https://docs.anaconda.com/anaconda/install/"

to create a virtual environment from anaconda :
" conda create -n nameofenvironment python=3.6.7"  // this project is made in python=3.6.7 version
                                                   // for me nameofmyenv is faceblur 
 My directory path is c:/mukund/desktop/python/day7/
 
 to activate virtual environment:
 "conda activate faceblur" 
 
 to import cv2 we need to first install opencv in our environment . to do that : 
 "pip install opencv-python opencv-contrib-python"
 numpy is already included in opencv . you don't need to install that too .
 
 recheck if all modules is installed correctly. 
 
 now ,
        just write code . i saved it as blur.py.
        
 make sure that environment is activated . 
 -->to run this code type :  "python blur.py gaussian" or "python blur.py pixelate"
 ## make sure your webcam is working.
 ## make sure that you put harcascade file in the same directory. 
 
IN THIS PROJECT , 
their is two type of blur option 
(1). Gaussian 
(2). Pixelate
