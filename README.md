# Face-Recognition-Attendance-Monitoring-System

## Approach:
1. look at a picture and find all the faces in it.
2. Focus on each face and understand that even if a face is 
   turned in a different direction or in bad lighting, it is still the same person.
3. Pick out unique features of the face that you can use to tell it apart 
   from other people— like how big the eyes are, how long the face is, etc.
4. Compare the unique features of that face to all the people you already know 
   to determine the person’s name.


## Techniques used:
a.) HOG (Histogram of Oriented Gradients) algorithm for finding faces.
b.) Generate Facial landmarks for faces posing and projected at different angles.
c.) 128 different measurements of a image generated. using these measurements 
	we can define a person, and differentiate b/w people as well.
d.) last step to differentiate them and  get names (used linear SVM classifier.)
