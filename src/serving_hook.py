import logging
import pickle

import cv2
import imageio
import io
import numpy as np
from openvino import inference_engine as ie
from PIL import Image
import six

import kuberlab_openvino as ko
from align import detect_face


LOG = logging.getLogger(__name__)
PARAMS = {
    'device': 'CPU',
    'align_model_dir': 'openvino-cpu',
    'resolutions': '26x37,37x52,52x74,145x206',
    'classifier': '',
    'threshold': [0.6, 0.7, 0.7]
}
width = 640
height = 480
net_loaded = False
pnets = None
rnet = None
onet = None
model = None
class_names = None


def init_hook(**kwargs):
    global PARAMS
    PARAMS.update(kwargs)

    if not isinstance(PARAMS['threshold'], list):
        PARAMS['threshold'] = [
            float(x) for x in PARAMS['threshold'].split(',')
        ]


def net_filenames(dir, net_name):
    base_name = '{}/{}'.format(dir, net_name)
    xml_name = base_name + '.xml'
    bin_name = base_name + '.bin'
    return xml_name, bin_name


class OpenVINONet(object):
    def __init__(self, plugin, net):
        self.exec_net = plugin.load(net)
        self.outputs = net.outputs
        self.input = list(net.inputs.keys())[0]

    def __call__(self, img):
        output = self.exec_net.infer({self.input: img})
        out = [output[x] for x in self.outputs]
        if len(out) == 1:
            return out[0]
        else:
            return out


def load_nets(**kwargs):
    global pnets
    global rnet
    global onet

    plugin = kwargs.get('plugin')
    model_dir = PARAMS.get('align_model_dir')

    LOG.info('Load PNET')

    pnets_proxy = []
    for r in ko.parse_resolutions(PARAMS['resolutions']):
        p = ko.PNetHandler(plugin, r[0], r[1])
        pnets_proxy.append(p.proxy())

    LOG.info('Load RNET')
    net = ie.IENetwork.from_ir(*net_filenames(model_dir, 'rnet'))
    rnet_proxy = OpenVINONet(plugin, net)

    LOG.info('Load ONET')

    net = ie.IENetwork.from_ir(*net_filenames(model_dir, 'onet'))
    onet_proxy = OpenVINONet(plugin, net)
    onet_input_name = list(net.inputs.keys())[0]
    onet_batch_size = net.inputs[onet_input_name][0]
    LOG.info('ONET_BATCH_SIZE = {}'.format(onet_batch_size))

    pnets, rnet, onet = detect_face.create_openvino_mtcnn(
        pnets_proxy, rnet_proxy, onet_proxy, onet_batch_size
    )

    LOG.info('Load classifier')
    with open(PARAMS['classifier'], 'rb') as f:
        global model
        global class_names
        opts = {'file': f}
        if six.PY3:
            opts['encoding'] = 'latin1'
        (model, class_names) = pickle.load(**opts)


def preprocess(inputs, ctx, **kwargs):
    global net_loaded
    if not net_loaded:
        load_nets(**kwargs)
        net_loaded = True

    image = inputs.get('input')
    if image is None:
        raise RuntimeError('Missing "input" key in inputs. Provide an image in "input" key')

    if isinstance(image[0], (six.string_types, bytes)):
        image = imageio.imread(image[0])

        rgba_image = Image.fromarray(image)
        image = rgba_image.convert('RGB')

    if image.shape[0] > height or image.shape[1] > width:
        frame = cv2.resize(
            image, (width, height), interpolation=cv2.INTER_AREA
        )
        scaled = True
    else:
        frame = image
        scaled = False

    bounding_boxes, _ = detect_face.detect_face_openvino(
        frame, pnets, rnet, onet, PARAMS['threshold']
    )
    ctx.scaled = scaled
    ctx.bounding_boxes = bounding_boxes
    ctx.frame = frame

    imgs = ko.get_images(frame, bounding_boxes)
    imgs = np.stack(imgs).transpose([0, 3, 1, 2])
    model_input = list(kwargs['model_inputs'].keys())[0]
    return {model_input: imgs}


def postprocess(outputs, ctx, **kwargs):
    facenet_output = list(outputs.values())[0]
    LOG.info('output shape = {}'.format(facenet_output.shape))

    labels = []

    for img_idx, item_output in enumerate(facenet_output):
        output = item_output.reshape(1, model.shape_fit_[1])
        predictions = model.predict_proba(output)

        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[
            np.arange(len(best_class_indices)),
            best_class_indices
        ]

        for i in range(len(best_class_indices)):
            bb = ctx.bounding_boxes[img_idx].astype(int)
            text = '%.1f%% %s' % (
                best_class_probabilities[i] * 100,
                class_names[best_class_indices[i]]
            )
            labels.append({
                'label': text,
                'left': bb[0],
                'top': bb[1] - 5
            })
            # DEBUG
            LOG.info('%4d  %s: %.3f' % (
                i,
                class_names[best_class_indices[i]],
                best_class_probabilities[i])
            )

    ko.add_overlays(ctx.frame, ctx.bounding_boxes, 0, labels=labels)
    image_bytes = io.BytesIO()
    imageio.imsave(image_bytes, ctx.frame, '.png')
    return {
        'output': image_bytes.getvalue(),
        'boxes': ctx.bounding_boxes,
        # 'labels': np.array(labels)
    }
