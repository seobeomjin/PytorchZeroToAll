import io 
import json 

from torchvision import models 
import torchvision.transforms as transforms 
from PIL import Image 
from flask import Flask, jsonify, request 

#imagenet class label load 
imagenet_class_index = json.load(open('C:/jupyter_devel/Pytorch_Tutorial/Pytorch_Tutorial_Official/_static/imagenet_class_index.json'))
#model load
model = models.densenet121(pretrained=True)
model.eval()

app = Flask(__name__)

# input data preprocessing 
def trasnform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485,0.456,0.406],
                                            [0.229,0.224,0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# input data prediction 
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.route('/')
def hello():
    return "Hello, Noah!"

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file'] #get fule 
        img_bytes = file.read() #change to bytes 
        class_id, class_name = get_prediction(image_bytes=img_bytes) #get predicted value 
        return jsonify({"class_id" : class_id , "class_name": class_name}) 
        
if __name__ == '__main__': 
    app.run()
