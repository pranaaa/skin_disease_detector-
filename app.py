from flask import Flask, render_template, request
import cv2
import csv
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
app=Flask(__name__)
@app.route('/',methods=['GET'])
def main_route():
    print("")
    return render_template('index.html')

@app.route('/index.html',methods=['GET'])
def index():
    
    return render_template('index.html')

@app.route('/about.html',methods=['GET'])
def about():
    
    return render_template('about.html')

@app.route('/contact.html',methods=['GET'])
def contact():
    
    return render_template('contact.html')

@app.route('/product.html',methods=['GET'])
def product():
    
    return render_template('product.html')

@app.route('/results.html',methods=['GET'])
def results():
    
    return render_template('results.html')

@app.route('/results.html',methods=['POST'])
def predict_image_classification_sample():
    model = load_model('/Users/pranathiprabhala/Desktop/models/my_model.h5') 
    filename=request.files["fileipt"].read()
    nparr = np.fromstring(filename, np.uint8)

   
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224,224))
    img = preprocess_input(np.array([img]))  
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    index_to_class = {}

    with open('label_mapping.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            index_to_class[int(row['Index'])] = row['Class Name']
    col=index_to_class[predicted_class_index]
    confidence = np.max(predictions)
    d=col
    return render_template('results.html',disease=d,confidence=confidence)
print("starting..")

if __name__=='__main__':
    app.run(port=3000,debug=True)

