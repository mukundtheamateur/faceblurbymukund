# faceblurbymukund
## THIS is an opencv PROJECT , 
## their are two types of blur option 
(1).Gaussian 
(2).Pixelate

#you can do this project without creating an environment but it is good practice to create one , so i did this project in an environment: 

--> IN THIS PROJECT WE WILL USE ANACONDA FOR CREATING A VIRTUAL ENVIRONMENT. 

Editor i used : visual studio code , to install click here "https://code.visualstudio.com/download"

i used command prompt to run my code. if you are using mac u can use terminal. 

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
 ## make sure you write argument to run code i.e  do not write "python blur.py" it will lead to error . write ""python blur.py gaussian"

