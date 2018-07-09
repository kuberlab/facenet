from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import argparse
import align.detect_face as detect_face
import tensorflow as tf
import numpy as np
import time
from scipy import misc


def get_parser():
    parser = argparse.ArgumentParser(
        description='Test movidious'
    )
    parser.add_argument(
        '--image',
        default=None,
        help='Image',
    )
    return parser


def get_size(scale):
    t = scale.split('x')
    return int(t[0]),int(t[1])
def imresample(img, h,w):
    im_data = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) #@UndefinedVariable
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
    face_crop_size=160
    face_crop_margin=32
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

    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()
    device = mvnc.Device(devices[0])
    device.open()


    print('Load RNET')

    with open('movidius/pnet-28x38.graph', mode='rb') as f:
        pgraphFileBuff = f.read()
    pnetGraph = mvnc.Graph("PNet Graph")
    pnetIn, pnetOut = pnetGraph.allocate_with_fifos(device, pgraphFileBuff)

    with open('movidius/rnet.graph', mode='rb') as f:
        rgraphFileBuff = f.read()
    rnetGraph = mvnc.Graph("RNet Graph")
    rnetIn, rnetOut = rnetGraph.allocate_with_fifos(device, rgraphFileBuff)

    with open('movidius/onet.graph', mode='rb') as f:
        ographFileBuff = f.read()
    onetGraph = mvnc.Graph("ONet Graph")
    onetIn, onetOut = onetGraph.allocate_with_fifos(device, ographFileBuff)

    with open('facenet.graph', mode='rb') as f:
        fgraphFileBuff = f.read()
    fGraph = mvnc.Graph("Face Graph")
    fifoIn, fifoOut = fGraph.allocate_with_fifos(device, fgraphFileBuff)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    #video_capture = cv2.VideoCapture(0)
    if args.image is None:
        from imutils.video import VideoStream
        from imutils.video import FPS
        vs = VideoStream(usePiCamera=True).start()
        time.sleep(1)
        fps = FPS().start()
    bounding_boxes = []
    with tf.Session() as  sess:
        def _pnet_proxy(img):
            pnetGraph.queue_inference_with_fifo_elem(pnetIn, pnetOut, img, 'pnet')
            output, userobj = pnetOut.read_elem()
            return output
        def _rnet_proxy(img):
            rnetGraph.queue_inference_with_fifo_elem(rnetIn, rnetOut, img, 'rnet')
            output, userobj = rnetOut.read_elem()
            return output
        def _onet_proxy(img):
            onetGraph.queue_inference_with_fifo_elem(onetIn, onetOut, img, 'onet')
            output, userobj = onetOut.read_elem()
            return output
        pnet,rnet,onet = detect_face.create_movidius_mtcnn(sess,'align',_pnet_proxy,_rnet_proxy,_onet_proxy)
        while True:
            # Capture frame-by-frame
            if args.image is None:
                frame = vs.read()
            else:
                frame = cv2.imread(args.image).astype(np.float32)
            #frame = cv2.resize(frame, (320, 320),interpolation=cv2.INTER_AREA)


            if (frame_count % frame_interval) == 0:
                bounding_boxes, _ = detect_face.movidius_detect_face(frame,pnet, rnet, onet,threshold)


                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count / (end_time - start_time))
                    start_time = time.time()
                    frame_count = 0

            if len(bounding_boxes)>0:
                #imgs = get_images(frame,bounding_boxes)
                imgs = []
                for i in imgs:
                    print(i.shape)
                    i = i.astype(np.float32)
                    fGraph.queue_inference_with_fifo_elem(fifoIn, fifoOut, i, 'user object')
                    output, userobj = fifoOut.read_elem()
                    print(output.shape)
                add_overlays(frame, bounding_boxes, frame_rate)

            frame_count += 1
            if args.image is None:
                cv2.imshow('Video', frame)
            else:
                print(bounding_boxes)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything is done, release the capture
    #video_capture.release()
    if args.image is None:
        fps.stop()
        vs.stop()
        cv2.destroyAllWindows()
    fifoIn.destroy()
    fifoOut.destroy()
    fGraph.destroy()
    rnetIn.destroy()
    rnetOut.destroy()
    rnetGraph.destroy()
    onetIn.destroy()
    onetOut.destroy()
    onetGraph.destroy()
    pnetIn.destroy()
    pnetOut.destroy()
    pnetGraph.destroy()
    device.close()
    print('Finished')


if __name__ == "__main__":
    main()