from __future__ import division, print_function
#import sys
import os
import cv2
#import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st
import onnxruntime as ort
import numpy as np
from keras.preprocessing.image import img_to_array
import pandas as pd

final_output_1 = 'happy'
ch = 'Movies'
df1 = pd.read_excel(rf"{ch}_Dataset.xlsx", sheet_name=final_output_1)
df3 = pd.read_excel(rf"{ch}_Dataset.xlsx", sheet_name=final_output_1)
recc = []
butt = 0
final_rat = 0

app = Flask(__name__)

def ratenow():
    global final_rat
    i=0

    GR_dict={0:(0,255,0),1:(0,0,255)}

    # model = tf.keras.models.load_model('final_model.h5')
    sess = ort.InferenceSession(r"C:\Users\Asus\Downloads\Emotflix\Emotflix-main\model.onnx", providers=["CPUExecutionProvider"])

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    output=[]
    rat_output=[]
    cap = cv2.VideoCapture(0)
    while (i<=30):
        ret, img = cap.read()
        # faces = face_cascade.detectMultiScale(img,1.05,5)
        if not ret:
                print("Ignoring empty frame")

        gray_image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

        for x,y,w,h in faces:

            # face_img = img[y:y+h,x:x+w] 

            # resized = cv2.resize(face_img,(224,224))
            # reshaped=resized.reshape(1, 224,224,3)

            roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
            roi_gray = cv2.resize(roi_gray,(48,48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255
            
            
            # sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# Set first argument of sess.run to None to use all model outputs in default order
# Input/output names are printed by the CLI and can be set with --rename-inputs and --rename-outputs
# If using the python API, names are determined from function arg names or TensorSpec names.
            predictions = sess.run(None, {"x": image_pixels})

            # predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)

            n_ratings = (1, 1, 1, 5, 2, 5, 3)
            predicted_ratings = n_ratings[max_index]
            rat_output.append(predicted_ratings)
            
            
            
            cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
            cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        i = i+1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27: 
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    
    final_rat = st.mode(rat_output)
    return render_template("ur_ratings.html",final_output=final_output1, final_rating = final_rat)

@app.route("/")
def home():
    return render_template("index1.html")
    
    
@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    global final_output_1
    i=0

    GR_dict={0:(0,255,0),1:(0,0,255)}

    # model = tf.keras.models.load_model('final_model.h5')
    sess = ort.InferenceSession(r"C:\Users\Asus\Downloads\Emotflix\Emotflix-main\model.onnx", providers=["CPUExecutionProvider"])

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    output=[]
    cap = cv2.VideoCapture(0)
    while (i<=30):
        ret, img = cap.read()
        # faces = face_cascade.detectMultiScale(img,1.05,5)
        if not ret:
                print("Ignoring empty frame")

        gray_image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

        for x,y,w,h in faces:

            # face_img = img[y:y+h,x:x+w] 

            # resized = cv2.resize(face_img,(224,224))
            # reshaped=resized.reshape(1, 224,224,3)

            roi_gray = gray_image[y-5:y+h+5,x-5:x+w+5]
            roi_gray = cv2.resize(roi_gray,(48,48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis = 0)
            image_pixels /= 255
            
            
            # sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# Set first argument of sess.run to None to use all model outputs in default order
# Input/output names are printed by the CLI and can be set with --rename-inputs and --rename-outputs
# If using the python API, names are determined from function arg names or TensorSpec names.
            predictions = sess.run(None, {"x": image_pixels})

            # predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)
            
            
            
            cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
            cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        i = i+1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27: 
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output_1 = st.mode(output)
    return render_template("buttons.html",final_output=final_output_1)


@app.route('/templates/buttons', methods = ['GET','POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/movies', methods = ['GET', 'POST'])
def movies():
    global ch
    global df1, df3
    global recc
    
    ch = 'Movies'
    print(ch)
    print(final_output_1)
    df1 = pd.read_excel(rf"{ch}_Dataset.xlsx", sheet_name=final_output_1)
    df3 = df1.sample(n=12)
    df3 = df3.sort_values(by=['Ratings'], ascending=False)
    df3

    recc = df3.to_dict('records')
    print(recc)
    return render_template("checking.html", RECC = recc)

@app.route('/songs', methods = ['GET', 'POST'])
def songs():
    global ch
    global df1, df3
    global recc
    
    ch = 'Songs'
    print(final_output_1)
    df1 = pd.read_excel(rf"{ch}_Dataset.xlsx", sheet_name=final_output_1)
    df3 = df1.sample(n=12)
    df3 = df3.sort_values(by=['Ratings'], ascending=False)
    df3

    recc = df3.to_dict('records')

    return render_template("checking.html", RECC = recc)

@app.route('/books', methods = ['GET', 'POST'])
def books():
    global ch
    global df1, df3
    global recc
    
    ch = 'Books'
    print(final_output_1)
    df1 = pd.read_excel(rf"{ch}_Dataset.xlsx", sheet_name=final_output_1)
    df3 = df1.sample(n=12)
    df3 = df3.sort_values(by=['Ratings'], ascending=False)
    df3

    recc = df3.to_dict('records')
    print(recc)
    return render_template("checking.html", RECC = recc)

@app.route('/proceed', methods = ['GET', 'POST'])
def proceed():
    recc[butt]['Ratings'] = final_rat

    idx = recc[butt]['id']

    row_index = df3.loc[df1['id'] == idx].index[0]
    df3.loc[row_index, 'Ratings'] = final_rat
    df1

    row_index = df1.loc[df1['id'] == idx].index[0]
    df1.loc[row_index, 'Ratings'] = final_rat
    df1

    def write_excel(filename,sheetname,dataframe):
        with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer: 
            workBook = writer.book
            try:
                workBook.remove(workBook[sheetname])
            except:
                print("Worksheet does not exist")
            finally:
                dataframe.to_excel(writer, sheet_name=sheetname,index=False)

    write_excel(rf"{ch}_Dataset.xlsx",final_output_1,df1)

    return render_template("checking.html", RECC = recc)

@app.route('/0', methods = ['GET', 'POST'])
def buttZero():
    global butt
    butt = 0
    return ratenow()

@app.route('/1', methods = ['GET', 'POST'])
def buttOne():
    global butt
    butt = 1
    return ratenow()

@app.route('/2', methods = ['GET', 'POST'])
def buttTwo():
    global butt
    butt = 2
    return ratenow()

@app.route('/3', methods = ['GET', 'POST'])
def buttThree():
    global butt
    butt = 3
    return ratenow()

@app.route('/4', methods = ['GET', 'POST'])
def buttFour():
    global butt
    butt = 4
    return ratenow()

@app.route('/5', methods = ['GET', 'POST'])
def buttFive():
    global butt
    butt = 5
    return ratenow()

@app.route('/6', methods = ['GET', 'POST'])
def buttSix():
    global butt
    butt = 6
    return ratenow()

@app.route('/7', methods = ['GET', 'POST'])
def buttSeven():
    global butt
    butt = 7
    return ratenow()

@app.route('/8', methods = ['GET', 'POST'])
def buttEight():
    global butt
    butt = 8
    return ratenow()

@app.route('/9', methods = ['GET', 'POST'])
def buttNine():
    global butt
    butt = 9
    return ratenow()

@app.route('/10', methods = ['GET', 'POST'])
def buttTen():
    global butt
    butt = 10
    return ratenow()

@app.route('/11', methods = ['GET', 'POST'])
def buttEleven():
    global butt
    butt = 11
    return ratenow()

@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")
    
if __name__ == "__main__":
    app.run(debug=True)