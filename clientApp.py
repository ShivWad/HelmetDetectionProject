from flask import Flask, json, request, jsonify, render_template
import os

from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import base64
from com_in_ineuron_ai_utils.utils import decodeImage, encodeImageIntoBase64
from research.obj import MultiClassObj

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


#@cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        modelPath = 'research/ssd_mobilenet_v1_coco_2017_11_17'
        self.objectDetection = MultiClassObj(self.filename, modelPath)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.files["file"]
    basepath = os.path.dirname(__file__)
    filename = image.filename
    file_path = os.path.join(basepath, 'static/uploads', secure_filename(image.filename))
    image.save(file_path)


    enc_img = encodeImageIntoBase64(file_path)

    decodeImage(enc_img, clApp.filename)

    result = clApp.objectDetection.getPrediction()

    result_img_64 = result[-1]['image']

    imgdata = base64.b64decode(result_img_64)

    with open("static/results/"+filename, 'wb+') as f:
        f.write(imgdata)
        f.close()
    path = ''
    d = ''
    try : 
        confid = result[0]['confidence']
        path = "static/results/"+filename
        d = {'image': path}
        return json.dumps(d)
        print('success')
    except Exception as e:
        path = "static/uploads/"+filename
        d = {'image': path}
        return json.dumps(d)
    


        

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    app.run(debug=True)