#%%
#Import Required Libraries
try:
    import cv2
    import numpy as np
    import face_recognition
    import warnings
    import os
    from datetime import datetime
    print("---Libraries Imported Successfully---")
    DeprecationWarning("ignore")
except:
    print("---Libraries Not Imported---")


# %%
#Change Directory
os.chdir('C:/Users/Ayush/Desktop/Face Recognition Attendance Monitoring System/Project Attendance')  #path where file is
warnings.filterwarnings("ignore")   #remove warnings

# %%
os.listdir()

# %%

path = "Students"  #path where my images are, in order to train my machine
images = []
classnames = []
mylist = os.listdir(path)
#print(mylist)

#taking name of the person from the image name in order to make some work easier.
for i in mylist:
    currentImage = cv2.imread(f'{path}/{i}')
    images.append(currentImage)
    classnames.append(os.path.splitext(i)[0])
#print(classnames)

#function to get 128 measurements of the face
def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

#marking attendance in csv file 
# concept is marking the attendance only once as soon as person is recognised
def attendance(name):
    with open('attendance.csv','r+',encoding='utf-8',errors='ignore',newline='') as f:
        datalist = f.readlines()
        namelist = []
        fieldnames = ['Name', 'Time']
        for line in datalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dt = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dt}')


#finding encodeings of known faces
encodelistknown = findencodings(images)

#initialising webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)  #to speed up the process resizing the image to one-forth of the original size
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    #we might find multiple faces so now finding face locations and then fnding encodings
    faceincurframe = face_recognition.face_locations(imgS)              #getting locations of the faces which we get from webcam
    encodecurframe = face_recognition.face_encodings(imgS,faceincurframe)           #getting encodings

    #now we will compare faces by iterating through all faces and check in known faces to get results
    for encodeface,faceloc in zip(encodecurframe,faceincurframe):
        matches = face_recognition.compare_faces(encodelistknown, encodeface)      #matching faces found in the frame with known faces
        facedis = face_recognition.face_distance(encodelistknown, encodeface)       #measuring distance -Gives a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. The distance tells you how similar the faces are.
        #print(facedis)
        matchindex = np.argmin(facedis)      #minimum the distance better the match

        if matches[matchindex]:
            name = classnames[matchindex].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4      # multiplying by 4 as earlier we resized our image to 1/4
            #putting results on the image
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0) ,2)   #Hollow rectangle around the face
            cv2.rectangle(img , (x1,y2-30), (x2,y2), (0,255,0) ,cv2.FILLED)  #Filled rectangle below hollow rectangle to print name on it
            cv2.putText(img,name,(x1+4,y2-4),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)    
            attendance(name)

    cv2.imshow("Webcam" , img)
    cv2.waitKey(1)

# %%
