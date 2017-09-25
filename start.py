from PIL import Image
from io import BytesIO
import face_recognition
import cherrypy
import base64
import numpy


def decode_image(encoded):
    bits = base64.b64decode(encoded)
    file = BytesIO(bits)
    return file


def encode_encoding(encoding):
    encoded = base64.b64encode(encoding)
    return encoded


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


class Server(object):
    @cherrypy.expose
    def index(self):
        return 'Hello!'

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def encode(self):
        if cherrypy.request.method == 'POST':
            data = cherrypy.request.json
            if data['image'] and len(data['image']) > 0:
                image = decode_image(data['image'])
                processed = process_image(image)
                encodings = get_encodings(processed)

                if len(encodings) > 0:
                    encoded = encode_encoding(encodings[0])
                    return {'encoding': encoded}
        return {'message': 'error'}

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def verify(self):
        if cherrypy.request.method == 'POST':
            data = cherrypy.request.json
            if data['image'] and data['encoding'] and len(data['image']) > 0 and len(data['encoding']) > 0:
                image = decode_image(data['image'])
                processed = process_image(image)
                unknown_encodings = get_encodings(processed)
                known_encoding = decode_encoding(data['encoding'])

                if len(unknown_encodings) > 0:
                    result = verify_identity(known_encoding, unknown_encodings[0])
                    return {'same': str(result)}
        return {'message': 'error'}


if __name__ == '__main__':
    cherrypy.config.update({'server.socket_host': '0.0.0.0',
                            'server.socket_port': 3000,
                            'log.screen': True,
                            'environment': 'production'})
    cherrypy.quickstart(Server(), '/')
