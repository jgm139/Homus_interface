import re
import numpy as np
import codecs
import os
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from sklearn.utils import graph
from model.load import init

app = Flask(__name__)

global model, graph
model, graph = init()

def convertImage(imgData1):
    img_str = re.search(b'base64,(.*)', imgData1).group(1)
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_out = os.path.join(THIS_FOLDER, 'out.png')
    with open(my_out, 'wb') as output:
        output.write(codecs.decode(img_str, 'base64'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    print("debug")
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_out = os.path.join(THIS_FOLDER, 'out.png')
    im = load_img(my_out, grayscale=True, target_size=[40, 40])  # this is a PIL image
    x = np.asarray(im).astype('float32') / 255
    print(x)
    x = x.reshape(1, 40, 40, 1)
    print("debug2")
    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        print("debug3")
        response = np.array_str(np.argmax(out, axis=1))
        return response


if __name__ == '__main__':
    app.run()
