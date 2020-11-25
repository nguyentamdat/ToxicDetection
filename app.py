from flask import Flask, request
from markupsafe import escape
from classifier import ToxicDetection
app = Flask(__name__)
toxicDetector = ToxicDetection()


@app.route('/', methods=['POST'])
def check():
    if request.is_json:
        data = request.get_json()
        return {'en': toxicDetector.predict(data.get('msg', '')), 'vn': toxicDetector.predict_vn(data.get('msg', ''))}
