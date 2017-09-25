from PIL import Image
from io import BytesIO
import face_recognition
import cherrypy
import base64


def get_encodings(file):
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model='cnn')
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings


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
            if data['image']:
                bits = base64.b64decode(data['image'])
                image = BytesIO(bits)
                processed = process_image(image)
                encodings = get_encodings(processed)

                if len(encodings) > 0:
                    print(encodings[0])
                    encoded = base64.b64encode(encodings[0])
                    # decoded = base64.b64decode(encoded)
                    # num = np.frombuffer(decoded)
                    # print(num)
                    return {'encoding': encoded}
            return {'key': 'value'}


if __name__ == '__main__':
    cherrypy.config.update({'server.socket_host': '0.0.0.0',
                            'server.socket_port': 3000,
                            'log.screen': True,
                            'environment': 'production'})
    cherrypy.quickstart(Server(), '/')
