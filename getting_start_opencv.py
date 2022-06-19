import cv2
import datetime
import os
import numpy as np

cap = cv2.VideoCapture(0)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# just capture the max value for device
#cap.set(3, 3000)
#cap.set(4, 4000)
# print(cap.get(3))
# print(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('ouput_teste.avi', fourcc, 20.0, (640, 480))
# model to get face
haar_model = os.path.join('/home/piero/Documentos',
                          #                          'haarcascade_frontalface_default.xml')
                          'haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier(haar_model)
# Glasses model
eye_cascade = cv2.CascadeClassifier(
    '/home/piero/Documentos/haarcascade_eye_tree_eyeglasses.xml')
glass_img = cv2.imread('/home/piero/Documentos/glasses.png')
print(cap.isOpened())
while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame', gray)
    # Put a dataset's frame
    dataset = str(datetime.datetime.now())
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, dataset, (10, 50),
                        font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.10, minNeighbors=6)
    centers = []
    try:
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (127, 0, 255), 1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                #                cv2.rectangle(roi_color, (ex, ey),
                #                              (ex+ew, ey+eh), (255, 255, 0), 2)
                centers.append(
                    (x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh)))
            if len(centers) > 0:
                # change the given value of 2.15 according to the size of the detected face
                glasses_width = 2.16 * abs(centers[1][0] - centers[0][0])
                overlay_img = np.ones(frame.shape, np.uint8) * 255
                h, w = glass_img.shape[:2]
                scaling_factor = glasses_width / w

                overlay_glasses = cv2.resize(
                    glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

                x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]

                # The x and y variables below depend upon the size of the detected face.
                x -= 0.26 * overlay_glasses.shape[1]
                y += 1.05 * overlay_glasses.shape[0]

                # Slice the height, width of the overlay image.
                h, w = overlay_glasses.shape[:2]
                overlay_img[int(y):int(y + h), int(x)                            :int(x + w)] = overlay_glasses

                # Create a mask and generate it's inverse.
                gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(
                    gray_glasses, 110, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                temp = cv2.bitwise_and(frame, frame, mask=mask)

                temp2 = cv2.bitwise_and(
                    overlay_img, overlay_img, mask=mask_inv)
                frame = cv2.add(temp, temp2)
    except:
        pass
        continue

    cv2.imshow('frame', frame)
    # RECORD ON IT!
    # out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# face_cascade = cv2.CascadeClassifier(haar_model)
# image = cv2.imread('/home/piero/Documentos/Piero/rapaziada no uruguai.jpg')
# image = cv2.resize(image, (800, 533))
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(
#     gray_image, scaleFactor=1.10, minNeighbors=0)
# for x, y, w, h in faces:
#     image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
#     cv2.imshow("Face Detector", image)
#     k = cv2.waitKey(2000)
# cv2.destroyAllWindows()
