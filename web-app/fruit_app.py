import base64
import os
import sys
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from gevent.pywsgi import WSGIServer


# Import Mask RCNN
from fruit_detect import make_detection


# Declare a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


print('Model loaded. Start serving...check: http://127.0.0.1:5000/')


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
        # print(f)
        basepath = os.path.dirname(__file__)
        image_path = os.path.join(
            basepath, 'rawImages', "testImage.png")
        f.save(image_path)

        # Make detection
        fruit_count = make_detection(image_path)

        # composing response
        response = {}
        with open('./maskedImages/result.png', 'rb') as resultImg:
            response['image'] = str(base64.b64encode(resultImg.read()))
            print(response['image'])
            response['code'] = 200
            response['count'] = fruit_count
        return jsonify(response)

    return None


if __name__ == '__main__':
    # app.run(port=5000, threaded=False)
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
