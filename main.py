# >pip install opencv-contrib-python
import cv2
# importing_haarcascade_classifiers
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# face_detection_function
def face_detection(image):
    # grascaling_image_passed
    if ret is True:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detecting_faces
        faces = face_classifier.detectMultiScale(gray, 1.6, 7)

        # drawing_face_rectangle
        for (x, y, w, h) in faces:
            # draw_rectangle_around_face
            cv2.rectangle(image, (x, y), (x + w, y + h), (127, 100, 255), 2)
        # for cropping
        for (x, y, w, h) in faces:
            cropped_face = image[y:y + h, x:x + w]

            return cropped_face

    # returning_image_with_rectangles
    return image


# capturing_video_from_webcam
cap = cv2.VideoCapture(0)
count = 0
while True:
    # reading_from_camera
    ret, frame = cap.read()
    count += 1

    face = cv2.resize(face_detection(frame), (200, 200))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)


    cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Face Cropper', face)

    # if_enter_pressed_then_exit
    if cv2.waitKey(1) == 13 or count == 100:
        break

# releasing_camera
cap.release()
# destroying_window
cv2.destroyAllWindows()

