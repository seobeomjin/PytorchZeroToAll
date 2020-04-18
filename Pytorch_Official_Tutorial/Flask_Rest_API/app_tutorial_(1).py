# 1
# API Definition 
# define our API endpoints, the request and response types. 
# {"class_id": "n02124075", "class_name": "Egyptian_cat"}

# 2
#Dependencies 
# pip install Flask==1.0.3 torchvision-0.3.0 
# But I do this on anaoncda virtual env 

# 3 
#Simple Web Server 
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World'

#decorator 
# 1. decorator function definition (probably it is predefined )
# 2. decorator get parameter which would be wrapped function
######################## eg ########################  
#import datetime
#
#
#def datetime_decorator(func):
#        def decorated():
#                print datetime.datetime.now()
#                func()
#                print datetime.datetime.now()
#        return decorated
#
#@datetime_decorator
#def main_function_1():
#        print "MAIN FUNCTION 1 START"
######################## eg ########################

# 3
# Save the above snippet in a file called "app.py" and you can now run a Flask development server
# FLASK_ENV=development FLASK_APP=app.py flask run 
##############
# help )https://abndistro.com/post/2019/01/20/using-flask-to-deploy-predictive-models/ 
# I got help from this site. In this site, there are explanations which how to deploy flask Apps using anaconda on windows 
############## 

# 4 (update ver. later)
# We will make slight changes to the above snippet, so that it suits our API definition
# update the endpoint path to /predict. Since the image files will be sent via HTTP POST requests, 
# we will update it so that it also accepts only POST requests
###
#@app.route('/predict',methods=['POST'])
#def predict():
#    return 'Hello World'
###

# 5 (update ver. later)
# also change the response type, so that it returns a JSON response containing ImageNet class id and name 
#@app.route('/predict',methods=['POST'])
#def predict():
#    return jsonify({'class_id':'IMGAE_NET_XXX','class_name':'Cat'})

# 6
# Inference code 
# there are 2 part 
# one part is where we prepare the image so that is can be fed to DenseNet  
# and next part is where we write the code to get the actual prediction from the model 


# 6-1
# Preparging the image 
# DenseNet model requires the image to be of 3 chaennel RGB image of size 224 x 224 
# also normalize the image tensor with required mean and standard deviation values. 

import io 

import torchvision.transforms as transforms 
from PIL import Image 

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
    # In-memory binary streams are also available as BytesIO objects
    #unsqueeze(0) >> it means adding a dimension in 0 position.  >> add dimension in the where parameter means   
    #squeeze() >> delete the dimension where element is '1'  

with open('C:/jupyter_devel/Pytorch_Tutorial/Pytorch_Tutorial_Official/test_img/test_cat.jpg','rb') as f:
    image_bytes = f.read()
    tensor = transform_image(image_bytes=image_bytes)
    print(tensor)


# 6-2 
# Prediction 
from torchvision import models 

# Make sure to pass 'pretrained' as 'True' to use pretrained weights 
model = models.densenet121(pretrained=True)
# Since we are using our model only for inference, switch to 'eval' mode 
model.eval() 

#def get_prediction(image_bytes):
#    tensor = transform_image(image_bytes=image_bytes)
#    outputs = model.forward(tensor)
#    _, y_hat = outputs.max(1)
#    return y_hat 
# y_hat will contain the index of the predicted class id 
# However, we need a human readable class name. For that we need a class id to name mapping. 
# Down load "imagenet_class_index.json" file and remember where 
# This file contains the mapping of ImageNet class id to ImageNet class name. 
# We will load this JSON file and get the class name of the predicted index

import json 
imagenet_class_index = json.load(open('C:/jupyter_devel/Pytorch_Tutorial/Pytorch_Tutorial_Official/_static/imagenet_class_index.json'))

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _ , y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

with open('C:/jupyter_devel/Pytorch_Tutorial/Pytorch_Tutorial_Official/test_img/test_cat.jpg','rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))
    # ['n02124075', 'Egyptian_cat'] 
    # first item is ImageNet class id , second item is the human readable name

"""
Note ::
#    Did you notice that ``model`` variable is not part of ``get_prediction``
#    method? Or why is model a global variable? Loading a model can be an
#    expensive operation in terms of memory and compute. If we loaded the model in the
#    ``get_prediction`` method, then it would get unnecessarily loaded every
#    time the method is called. Since, we are building a web server, there
#    could be thousands of requests per second, we should not waste time
#    redundantly loading the model for every inference. So, we keep the model
#    loaded in memory just once. In
#    production systems, it's necessary to be efficient about your use of
#    compute to be able to serve requests at scale, so you should generally
#    load your model before serving requests.
"""

# 7 
# Integrating the model in out API Server 
# we will add out model to out Flask API server 
# Since out API srver is supposed to take an image file, 
# we will update our predict method to read files from requests 

from flask import request 

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        #we will get the file from the request 
        file = request.files['file']    
        #convert that to bytes 
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id' : class_id, 'class_name' : class_name})

if __name__ == '__main__':
    app.run()




