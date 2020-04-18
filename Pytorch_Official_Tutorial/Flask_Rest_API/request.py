
import requests
resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('C:/jupyter_devel/Pytorch_Tutorial/Pytorch_Tutorial_Official/test_img/test_cat.jpg','rb')})
