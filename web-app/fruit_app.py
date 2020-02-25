import base64
import os
import sys
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from timeit import default_timer as timer

from fruit_detect import mask_detect,yolo_detect


# Declare a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


print('Model loaded. Start serving...check: http://127.0.0.1:5000/')


@app.route('/model', methods=['POST'])
def get_model():
    global fruit_model
    if request.method == 'POST':
        model = request.json['model']
        # print(model)
        fruit_model = model
    return "OK"

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('Error: No image file')
            res_dict = {}
            res_dict['code'] = 404
            return jsonify(res_dict)
        else:
            f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        # image_path = os.path.join(
        #     basepath, 'rawImages', secure_filename(f.filename))
        image_path = os.path.join(
            basepath, 'rawImages', "testImage.png")
        f.save(image_path)


        model = fruit_model
        print("model used:"+model)
        start = timer()
        if model=="mask":
            fruit_count = mask_detect(image_path)
        else:
            fruit_count = yolo_detect(image_path)
        end =timer()
        processing_time=round((end - start),5)

        # composing response
        response = {}
        with open('./maskedImages/result.png', 'rb') as res_img:
            response['image'] = str(base64.b64encode(res_img.read()))
            response['code'] = 200
            response['count'] = fruit_count
            response['time'] = processing_time
        return jsonify(response)

    return None


if __name__ == '__main__':

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
