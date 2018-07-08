from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import argparse

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
    return parser


def get_size(scale):
    t = scale.split('x')
    return int(t[0]),int(t[1])
def imresample(img, h,w):
    im_data = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) #@UndefinedVariable
    return im_data

def main():
    parser = get_parser()
    args = parser.parse_args()
    devices = mvnc.enumerate_devices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    device = mvnc.Device(devices[0])
    device.open()
    graph = mvnc.Graph('graph')
    print('Load movidius/pnet-{}.graph'.format(args.size))
    with open('movidius/pnet-{}.graph'.format(args.size), mode='rb') as f:
        graphFileBuff = f.read()
    fifoIn, fifoOut = graph.allocate_with_fifos(device, graphFileBuff)
    img = cv2.imread(args.image).astype(np.float32)
    print(img.shape)
    h,w = get_size(args.size)
    im_data = imresample(img,h,w)
    im_data = (im_data-127.5)*0.0078125
    print(im_data.shape)
    #img_y = np.transpose(im_data, (1,0,2))
    #print(img_y.shape)
    img_x = im_data.reshape((1,h,w,3))
    img_y = np.transpose(img_x, (0,2,1,3))
    print('Start download to NCS...')
    graph.queue_inference_with_fifo_elem(fifoIn, fifoOut, img_y, 'user object')
    output, userobj = fifoOut.read_elem()
    print(output.shape)
    ##output = output.reshape((int(h/2)-5, int(w/2)-5,6))
    print(output)
    print(userobj)
    fifoIn.destroy()
    fifoOut.destroy()
    graph.destroy()
    device.close()
    print('Finished')

if __name__ == "__main__":
    main()