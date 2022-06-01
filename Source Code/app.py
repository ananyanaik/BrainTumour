import os
from os.path import join, dirname, realpath
import numpy as np  
from PIL import Image
#keras
from keras.models import load_model
from keras.preprocessing import image
import cv2
from keras.applications import vgg16
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.optimizers
from keras.applications import vgg16

# Flask utils
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request,redirect

# Define a flask app
new_model = load_model('Source Code/brain_tumor_model.h5')

app = Flask(__name__)
vgg16_model = vgg16.VGG16()
model1 = Sequential()
#for layer in vgg16_model.layers:
for layer in vgg16_model.layers[:-1]: #model1.layers.pop() not working, so -1
    model1.add(layer) # convert to sequantial model

for layer in model1.layers:
    layer.trainable = False
    
model1.add(Dense(4,activation = 'softmax'))
model1.load_weights("Source Code/weights.h5")

IMG_SIZE = 224
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
def predict(file):
    prediction = model1.predict([prepare(file)])
    #output = CATEGORIES[int(prediction[0][0])]
    if np.argmax(prediction)==0:    
        output="glioma"
    if np.argmax(prediction)==1:
        output="meningioma"
    if np.argmax(prediction)==2:
        output="no_tumor"
    if np.argmax(prediction)==3:
        output="pituitary" 
    return output
# Model saved with Keras model.save()
@app.route('/',methods=['GET'])
@app.route('/index',methods=['GET'])
def index():
	# Main page
    return render_template('index.html')

# @app.route('/',methods=['GET'])
# def upload():
# 	# Main page
#     return render_template('index.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/upload')
def upload():
	return render_template('upload.html')

@app.route('/uploader',methods=['POST','GET'])
def uploader():
	if request.method == 'POST':
		# Get the file from post request
		f = request.files['file']

		path = os.path.dirname(__file__)
		basepath=os.path.normpath(path)

		file_path=os.path.join(basepath,'uploads',secure_filename(f.filename)).replace("\\","/")
		f.save(file_path)
		img5 = image.load_img(file_path , target_size=(240,240))
		images5 = image.img_to_array(img5)
		images5 = np.expand_dims(images5,axis=0)
		
				
		out = predict(file_path)
		print(out)
		prediction = new_model.predict(images5)
		
		

		#result = prediction

		if prediction <=0.5 :
		    #print("Tumor Not Detected")
		    #plt.title('No')
		    #return "The Person has no Brain Tumor"
			return render_template('no.html')
			# url = 'https://www.yahoo.com/'
			# return redirect(url, code=307)
		else:
		    #print("Tumor Detected")
		    #plt.title('Yes')
		    #return 'file uploaded succesfully'
			 #sql = 'https://index-report.netlify.app/'
			 #return redirect(sql, code=300)
			return render_template('Yes.html',t_type=out)

if __name__ == '__main__':
    app.run(debug=True)
