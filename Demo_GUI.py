import cv2
import skvideo
import numpy as np
import cv2
from extract_features_c3d import C3D
from preprocess_input import preprocess_input
import skvideo.io
import tensorflow as tf
from keras.models import Model



# loading the C3D model for spatiotemporal feature extracted
k = C3D(weights='sports1M')
# modyfying the output of the layer to extract 4096 vector
model1 = Model(inputs=k.input, outputs=k.get_layer('fc7').output)
# loding the trained regression model
model2 = tf.keras.models.load_model('my_model.h5')



def label_k(val):
    if val > 0.5:
        return " Anomaly detected"
    else:
        return "NORMAL"

# predict video using the base model and regresion model
def predict_vid(model1, model2, vid):
    k = preprocess_input(vid)
    k = model1.predict(k)
    k = model2.predict(k)
    label = label_k(k)
    return label + " "+ str(k[0][0]) + " Anomaly score"


# label frames based on anomaly
def label_frames(path):
    label_list = []
    kb = C3D(weights='sports1M')
    model1 = Model(inputs=kb.input, outputs=kb.get_layer('fc7').output)
    model2 = tf.keras.models.load_model('my_model.h5')
    vid = skvideo.io.vread(path)
    (num_frames,_,_,_) = vid.shape
    num_cut = num_frames//16
    cut_list = [x*16 for x in range(1,num_cut)]
    begin = 0
    for x in cut_list:
        k = vid[begin:x]
        begin = x
        label = predict_vid(model1,model2,k)
        label_list.append(label)


    return (label_list, cut_list)



(label_list, cut_list )= label_frames('/home/eric/PycharmProjects/video_anomaly_detection/deData/NUDE/10.avi')

print(label_list)



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('/home/eric/PycharmProjects/video_anomaly_detection/deData/NUDE/10.avi')


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
c = 1
# Read until video is completed
label = label_list[0]
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
       key = cv2.waitKey(500)
       if c in cut_list:
           index = cut_list.index(c)
           label = label_list[index]
           cv2.putText(frame, label, org=(10, 200),fontFace=1,fontScale=2,color=(255,255,255))
        # Display the resulting frame
       cap.set(cv2.CAP_PROP_FPS, 180)
       cv2.namedWindow('Video Player')
       cv2.resizeWindow('Video Player', 600, 600)
       cv2.imshow('Video Player', frame)
       c = c + 1

        # Press Q on keyboard to  exit
       if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()



