import pickle

import cv2
import argparse
import align.detect_face as detect_face
import tensorflow as tf
import numpy as np
import time
from scipy import misc

import facenet


def get_parser():
    parser = argparse.ArgumentParser(
        description='Test movidious'
    )
    parser.add_argument(
        '--size',
        default='360x480',
        help='Image size',
    )
    parser.add_argument(
        '--image',
        default='test.png',
        help='Image',
    )
    parser.add_argument(
        '--classifier',
        help='Path to classifier file.',
    )
    parser.add_argument('--tf-graph-path')
    return parser


def get_size(scale):
    t = scale.split('x')
    return int(t[0]),int(t[1])


def imresample(img, h,w):
    im_data = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)  # @UndefinedVariable
    return im_data


def add_overlays(frame, boxes, frame_rate):
    if boxes is not None:
        for face in boxes:
            face_bb = face.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def get_images(image, bounding_boxes):
    face_crop_size = 160
    face_crop_margin = 32
    images = []

    for bb in bounding_boxes:
        bounding_box = np.zeros(4, dtype=np.int32)
        img_size = np.asarray(image.shape)[0:2]
        bounding_box[0] = np.maximum(bb[0] - face_crop_margin / 2, 0)
        bounding_box[1] = np.maximum(bb[1] - face_crop_margin / 2, 0)
        bounding_box[2] = np.minimum(bb[2] + face_crop_margin / 2, img_size[1])
        bounding_box[3] = np.minimum(bb[3] + face_crop_margin / 2, img_size[0])
        cropped = image[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
        image = misc.imresize(cropped, (face_crop_size, face_crop_size), interp='bilinear')
        images.append(image)
    return images


def main():
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0
    start_time = time.time()

    parser = get_parser()
    args = parser.parse_args()

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    if bool(args.classifier) ^ bool(args.tf_graph_path):
        raise ValueError('tf_graph path and classifier must be filled.')
    use_classifier = False

    if args.classifier and args.tf_graph_path:
        use_classifier = True

    video_capture = cv2.VideoCapture(0)

    if use_classifier:
        with open(args.classifier, 'rb') as f:
            (model, class_names) = pickle.load(f)

    bounding_boxes = []
    with tf.Session() as sess:
        pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')
        if use_classifier:
            facenet.load_model(args.tf_graph_path)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        try:
            while True:
                ret, frame = video_capture.read()
                #frame = cv2.imread(args.image).astype(np.float32)
                frame = cv2.resize(frame, (640, 480),interpolation=cv2.INTER_AREA)

                if (frame_count % frame_interval) == 0:
                    bounding_boxes, _ = detect_face.detect_face(
                        frame, minsize, pnet, rnet, onet, threshold, factor
                    )
                    # Check our current fps
                    end_time = time.time()
                    if (end_time - start_time) > fps_display_interval:
                        frame_rate = int(frame_count / (end_time - start_time))
                        start_time = time.time()
                        frame_count = 0

                if len(bounding_boxes) > 0:
                    if use_classifier:
                        imgs = get_images(frame, bounding_boxes)
                        for img_idx, img in enumerate(imgs):
                            img = img.astype(np.float32)
                            feed_dict = {
                                images_placeholder: [img],
                                phase_train_placeholder: False
                            }
                            embedding = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(embedding)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                            for i in range(len(best_class_indices)):
                                bb = bounding_boxes[img_idx].astype(int)
                                text = '%.1f%% %s' % (best_class_probabilities[i] * 100, class_names[best_class_indices[i]])
                                cv2.putText(
                                    frame, text, (bb[0], bb[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                    thickness=1, lineType=2
                                )
                                # print('%4d  %s: %.3f' % (
                                #     i,
                                #     class_names[best_class_indices[i]],
                                #     best_class_probabilities[i])
                                # )

                    add_overlays(frame, bounding_boxes, frame_rate)

                frame_count += 1
                cv2.imshow('Video', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except (KeyboardInterrupt, SystemExit, Exception) as e:
            print('Caught %s: %s' % (e.__class__.__name__, e))

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print('Finished')


if __name__ == "__main__":
    main()