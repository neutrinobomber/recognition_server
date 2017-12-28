import cv2
from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
import face_recognition
from face_recognition import face_locations
from face_recognition.cli import image_files_in_folder


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def train(train_dir, model_save_path='', n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model of disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically
    if not specified.
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    x = []
    y = []
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:
                    print('image {} not fit for training: {}'.format(
                        img_path,
                        'didn\'t find a face' if len(faces_bboxes) < 1 else 'found more than one face'))
                continue
            x.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(x))))
        if verbose:
            print('Chose n_neighbors automatically as:', n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(x, y)

    if model_save_path != '':
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)
    return knn_clf


def predict(x_img_path, knn_clf=None, model_save_path='', dist_thresh=.5):
    """
    recognizes faces in given image, based on a trained knn classifier

    :param x_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_save_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must
    be knn_clf.
    :param dist_thresh: (optional) distance threshold in knn classification. the larger it is, the more chance of
    misclassifying an unknown person to a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'N/A' will be passed.
    """

    if not isfile(x_img_path) or splitext(x_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception('invalid image path: {}'.format(x_img_path))

    if knn_clf is None and model_save_path == '':
        raise Exception('must supply knn classifier either thourgh knn_clf or model_save_path')

    if knn_clf is None:
        with open(model_save_path, 'rb') as f:
            knn_clf = pickle.load(f)

    x_img = face_recognition.load_image_file(x_img_path)
    x_faces_loc = face_locations(x_img)
    if len(x_faces_loc) == 0:
        return []

    faces_encodings = face_recognition.face_encodings(x_img, known_face_locations=x_faces_loc)

    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)

    is_recognized = [closest_distances[0][i][0] <= dist_thresh for i in range(len(x_faces_loc))]

    # predict classes and cull classifications that are not with high confidence
    return [(pred, loc) if rec else ("N/A", loc) for pred, loc, rec in zip(
        knn_clf.predict(faces_encodings),
        x_faces_loc, is_recognized)]


video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
user_image = face_recognition.load_image_file('images/user/user.jpg')
user_face_encoding = face_recognition.face_encodings(user_image)[0]

# Initialize some variables
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(small_frame, number_of_times_to_upsample=0, model='cnn')
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces([user_face_encoding], face_encoding)
            name = 'Unknown'

            if match[0]:
                name = 'User'

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
