from flask import Flask
from flask import request, jsonify
import face_recognition
from PIL import Image
from io import BytesIO
import base64
import numpy
import time
import threading
import urllib.request
import os


app = Flask(__name__)


def keep_alive():
    threading.Timer(300, keep_alive).start()
    try:
        urllib.request.urlopen(os.environ['URL'])
    except Exception as e:
        print(str(e))


def decode_image(encoded):
    bits = base64.b64decode(encoded)
    file = BytesIO(bits)
    return file


def encode_encoding(encoding):
    encoded = base64.b64encode(encoding)
    string = encoded.decode('utf-8')
    return string


def decode_encoding(encoding):
    decoded = base64.b64decode(encoding)
    result = numpy.frombuffer(decoded)
    return result


def get_encodings(file):
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model='cnn')
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings


def verify_identity(known_encoding, unknown_encoding):
    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    return results[0]


def process_image(image):
    size = 300, 300
    image = Image.open(image)
    image.thumbnail(size)
    new = BytesIO()
    image.save(new, 'jpeg')
    return new


@app.route('/')
def hello():
    return 'Hello!'


@app.route('/encode', methods=['POST'])
def encode():
    data = request.json
    if data['image'] and len(data['image']) > 0:
        image = decode_image(data['image'])
        processed = process_image(image)

        start = time.perf_counter()
        encodings = get_encodings(processed)
        print(time.perf_counter() - start)

        if len(encodings) > 0:
            encoded = encode_encoding(encodings[0])
            return jsonify({'encoding': encoded})

    return jsonify({'message': 'ivalid data'})


@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    if data['image'] and data['encoding'] and len(data['image']) > 0 and len(data['encoding']) > 0:
        image = decode_image(data['image'])
        processed = process_image(image)

        start = time.perf_counter()
        unknown_encodings = get_encodings(processed)
        known_encoding = decode_encoding(data['encoding'])
        print(time.perf_counter() - start)

        if len(unknown_encodings) > 0:
            result = verify_identity(known_encoding, unknown_encodings[0])
            return jsonify({'same': str(result)})

    return jsonify({'message': 'invalid data'})


if __name__ == "__main__":
    keep_alive()
    app.run(host='0.0.0.0', port=3000)
