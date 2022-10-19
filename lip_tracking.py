import dlib
import numpy as np
import skimage.transform
import cv2
from matplotlib import pyplot as plt

class lip_tracking:
    def detector_predictor():
        train = 'train'

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        return detector, predictor

    def face_detection(frames):
        detector, predictor = lip_tracking.detector_predictor()
        mouth_frames = []
        MOUTH_WIDTH = 112
        MOUTH_HEIGHT = 112

        try:
            for frame in frames:
                dets = detector(frame, 1)
                # shape = None
                for k, d in enumerate(dets):
                    shape = predictor(frame, d)
                    i = -1
                # if shape is None:
                #     return frames
                mouth_points = []
                for part in shape.parts():
                    i += 1
                    if i == 0:
                        face_left = (part.x, part.y)
                    if i == 16:
                        face_right = (part.x, part.y)
                    if i < 48: # 입 부분만 검출
                        continue
                    mouth_points.append((part.x,part.y))
                    np_mouth_points = np.array(mouth_points)

                mouth_centroid = np.mean(np_mouth_points, axis=0)

                # if normalize_ratio is None:
                #     mouth_left = np.min(np_mouth_points[:, :-1]) * (1.0 - HORIZONTAL_PAD)
                #     mouth_right = np.max(np_mouth_points[:, :-1]) * (1.0 + HORIZONTAL_PAD)

                #     normalize_ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)

                # new_img_shape = (int(frame.shape[0] * normalize_ratio), int(frame.shape[1] * normalize_ratio))
                # resized_img = skimage.transform.resize(frame, new_img_shape)


                # mouth_centroid_norm = mouth_centroid * normalize_ratio

                mouth_l = int(mouth_centroid[0] - MOUTH_WIDTH / 2)
                mouth_r = int(mouth_centroid[0] + MOUTH_WIDTH / 2)
                mouth_t = int(mouth_centroid[1] - MOUTH_HEIGHT / 2)
                mouth_b = int(mouth_centroid[1] + MOUTH_HEIGHT / 2)

                mouth_crop_image = frame[mouth_t:mouth_b, mouth_l:mouth_r]

                # mouth_crop_image = resized_img[mouth_t:mouth_b, mouth_l:mouth_r]
                mouth_crop_image = np.array(mouth_crop_image) 
                # mouth_crop_image =  mouth_crop_image.astype(np.float32)
                # mouth_crop_image = cv2.cvtColor(mouth_crop_image, cv2.COLOR_BGR2RGB)


                mouth_frames.append(mouth_crop_image)
        except Exception:
            pass
        return mouth_frames, mouth_l, mouth_r, mouth_t, mouth_b, face_left, face_right

# mp4_path = "Lip_Reading_in_the_wild/lipread_mp4"
# image_path = "Lip_Reading_in_the_wild/lipread_image"
# train = 'train'
# mp4_path, list_mp4 = load_path.load_path.total_path()
# save_frame.dataset(mp4_path, len(list_mp4), list_mp4, image_path, train)