"""
Importing necessary libraries
"""
import cv2,mtcnn
from mtcnn.mtcnn import MTCNN
import tensorflow
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')
classifier = load_model('mask_detector.model') #Classifier used to predict mask or not mask


#Initializing the front camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read() #It read the frames from the live camera
    fcs=[]
    preds=[]
    locs=[]
    cfd=[]

    """
    Now we will use either of Haar Cascade or mtcnn to detect the faces through 
    bounding boxes and use classifier to predict the faces
    """

    #For Haar Cascade
    hc = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    hcfaces = hc.detectMultiScale(frame,1.2)
    for (x,y,w,h) in hcfaces:
        x2=x+w
        y2=y+h
        fc=frame[y:y2, x:x2] #capturing only faces
        fc = cv2.resize(fc, (224, 224))
        fc = img_to_array(fc) #Converting face image to array
        fc = preprocess_input(fc)
        fcs.append(fc)
        locs.append((x, y, x2, y2))


    # #For mtcnn.
    # """
    # (It requires large computation power, but gives better accuracy for face detection than
    # Haar Cascade. Run only when u have high end laptop)
    # """
    # mt = MTCNN()
    # faces = mt.detect_faces(frame) #it returns dictionary containing bbox, confidence and
    # # points of eyes,nose
    #
    # for face in faces:
    #     x,y,w,h = face['box']
    #     confidence = face['confidence']
    #     x2=x+w
    #     y2=y+h
    #     fc=frame[y:y2, x:x2]
    #     fc= cv2.resize(fc, (224,224))
    #     fc=img_to_array(fc)
    #     fc= preprocess_input(fc)
    #     fcs.append(fc)
    #     locs.append((x, y, x2, y2))
    #     cfd.append(confidence)

    if len(fcs) > 0:
        fcs = np.array(fcs, dtype="float32") # Just ensuring that images are in array form
        preds = classifier.predict(fcs, batch_size=32)

    for (bbox, pred) in zip(locs, preds):
        (x, y, x2, y2) = bbox
        (m, nm) = pred
        label = "Mask" if m>nm else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # green if mask else red
        label = "{}: {:.2f}%".format(label, max(m, nm) * 100)

        cv2.putText(frame, label, (x, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2) #Putting text in live frames

        cv2.rectangle(frame, (x, y), (x2, y2), color, 3) #Putting rectangle of bbox in frames
        cv2.imshow('frame', frame)


    if (cv2.waitKey(20) == ord('q')) or (cv2.waitKey(20) == 27):
        # pressing 'q' or 'esc' keys destroys the window
        break

cap.release()
cv2.destroyAllWindows()

"""
The accuracy of the classifier can be increased by changing with new more accurate classifier
or fine tuning the current one again
"""