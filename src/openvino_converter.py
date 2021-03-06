import align.detect_face as df
import tensorflow as tf
import os
import numpy as np
import subprocess
import logging
import argparse
import datetime


def submit(params):
    if os.environ.get('PROJECT_ID', None):
        from mlboardclient.api import client
        client.update_task_info(params)


def push(name, dirame):
    if os.environ.get('PROJECT_ID', None):
        from mlboardclient.api import client
        timestamp = datetime.datetime.now().strftime('%s')
        if name is not None:
            version = '1.0.0-{}-{}'.format(name, timestamp)
        else:
            version = '1.0.0-{}'.format(timestamp)
        mlboard = client.Client()
        mlboard.model_upload('openvino-facenet', version, dirame)
        submit({'model': '{}:{}'.format('openvino-facenet', version)})
        logging.info("New model uploaded as 'openvino-facenet', version '%s'." % (version))


def conver_onet(dir, data_type='FP32', prefix=None, do_push=False):
    # Set batch size for conversion
    batch_size = 2

    out_dir = os.path.join(dir, "openvino")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dir = os.path.join(dir, "onet")
    if not os.path.exists(dir):
        os.mkdir(dir)

    tf.reset_default_graph()
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        data = tf.placeholder(tf.float32, (1, 48, 48, 3), 'input')
        logging.info("Load ONET graph")
        with tf.variable_scope('onet'):
            onet = df.ONetOpenVINO({'data': data})
            onet.load(os.path.join('align', 'det3.npy'), sess)

        logging.info("Create ONET output layer")

        onet_output0 = graph.get_tensor_by_name('onet/conv6-2/conv6-2:0')
        onet_output1 = graph.get_tensor_by_name('onet/conv6-3/conv6-3:0')
        onet_output2 = graph.get_tensor_by_name('onet/prob1:0')

        tf.identity(onet_output0, name='onet/output0')
        tf.identity(onet_output1, name='onet/output1')
        tf.identity(onet_output2, name='onet/output2')

        logging.info("Freeze ONET graph")

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), ['onet/output0', 'onet/output1', 'onet/output2']
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(os.path.join(dir, 'onet.pb'), "wb") as f:
            f.write(output_graph_def.SerializeToString())

        cmd = 'mo_tf.py --input_model {0}/onet.pb --output_dir {0} --data_type {1} --batch {2}'.format(
            dir, data_type, batch_size
        )
        logging.info('Compile: %s', cmd)
        result = subprocess.check_output(cmd, shell=True).decode()
        logging.info(result)
        if do_push:
            push('onet', out_dir)


def convert_rnet(dir, data_type='FP32', prefix=None, do_push=False):
    out_dir = os.path.join(dir, "openvino")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dir = os.path.join(dir, "rnet")
    if not os.path.exists(dir):
        os.mkdir(dir)
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        logging.info("Load RNET graph")
        data = tf.placeholder(tf.float32, (1, 24, 24, 3), 'input')
        with tf.variable_scope('rnet'):
            onet = df.RNetOpenVINO({'data': data})
            onet.load(os.path.join('align', 'det2.npy'), sess)
        logging.info("Create RNET output")

        rnet_output0 = graph.get_tensor_by_name('rnet/conv5-2/conv5-2:0')
        rnet_output1 = graph.get_tensor_by_name('rnet/prob1:0')

        tf.identity(rnet_output0, name='rnet/output0')
        tf.identity(rnet_output1, name='rnet/output1')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        logging.info("Freeze RNET graph")

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), ['rnet/output0', 'rnet/output1']
        )
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(os.path.join(dir, 'rnet.pb'), "wb") as f:
            f.write(output_graph_def.SerializeToString())

        cmd = 'mo_tf.py --input_model {0}/rnet.pb --output_dir {0} --data_type {1}'.format(dir, data_type)
        logging.info('Compile: %s', cmd)
        result = subprocess.check_output(cmd, shell=True).decode()
        logging.info(result)
        if do_push:
            push('rnet', out_dir)


def convert_pnet(dir, h, w, data_type='FP32', prefix=None):
    logging.info("Prepare PNET-{}x{} graph".format(h, w))
    out_dir = os.path.join(dir, "openvino")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dir = os.path.join(dir, 'pnet'.format(h, w))
    if not os.path.exists(dir):
        os.mkdir(dir)
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph, tf.Session() as sess:
        logging.info("Load PNET graph")
        data = tf.placeholder(tf.float32, (1, h, w, 3), 'input')
        with tf.variable_scope('pnet'):
            pnet = df.PNetOpenVINO({'data': data})
            pnet.load(os.path.join('align', 'det1.npy'), sess)
        logging.info("Create PNET output")
        pnet_output0 = graph.get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
        pnet_output1 = graph.get_tensor_by_name('pnet/prob1:0')

        tf.identity(pnet_output0, name='pnet/output0')
        tf.identity(pnet_output1, name='pnet/output1')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        logging.info("Freeze PNET graph")

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), ['pnet/output0', 'pnet/output1']
        )

        out_file = 'pnet_{}x{}.pb'.format(h, w)

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(os.path.join(dir, out_file), "wb") as f:
            f.write(output_graph_def.SerializeToString())

        cmd = 'mo_tf.py --input_model {0}/{1} --output_dir {0} --data_type {2}'.format(dir, out_file, data_type)
        logging.info('Compile: %s', cmd)
        result = subprocess.check_output(cmd, shell=True).decode()
        logging.info(result)


def prepare_pnet(dir, data_type='FP32', do_push=False, prefix=None):
    minsize = 20  # minimum size of face
    factor = 0.709  # scale factor
    factor_count = 0
    h = 480
    w = 680
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1
    # first stage
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        convert_pnet(dir, hs, ws, data_type=data_type, prefix=prefix)
    if do_push:
        out_dir = os.path.join(dir, "openvino")
        push('pnet', out_dir)


def convert_facenet(dir, frozen_graph_path, data_type='FP32', do_push=False):
    out_dir = os.path.join(dir, 'openvino')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dir = os.path.join(dir, "facenet")
    if not os.path.exists(dir):
        os.mkdir(dir)

    cmd = (
        'mo_tf.py --input_model {0} --freeze_placeholder_with_value'
        ' "phase_train->False" --data_type {1} --output_dir {2}'.format(
            frozen_graph_path, data_type, dir
        )
    )
    logging.info('Compile: %s', cmd)
    result = subprocess.check_output(cmd, shell=True).decode()
    logging.info(result)
    if do_push:
        push('facenet', out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--training_dir',
        help='Training dir',
        required=True,
    )

    parser.add_argument(
        '--target',
        help='Compile for target device',
        default='CPU',
        choices=['CPU', 'MYRIAD', 'MOVIDIUS']
    )
    parser.add_argument(
        '--onet',
        action='store_true',
        help='Build ONET'
    )
    parser.add_argument(
        '--pnet',
        action='store_true',
        help='Build PNET'
    )
    parser.add_argument(
        '--rnet',
        action='store_true',
        help='Build RNET'
    )
    parser.add_argument(
        '--facenet',
        action='store_true',
        help='Build FACENET'
    )
    parser.add_argument(
        '--do_push',
        action='store_true',
        help='Push model to catalog'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Convert all'
    )
    parser.add_argument(
        '--facenet_graph',
        help='Base model path'
    )
    parser.add_argument(
        '--output-size',
        type=int,
        default=512,
        help='Facenet model output size'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=160,
        help='Facenet model input size'
    )
    return parser.parse_args()


data_types = {
    'CPU': 'FP32',
    'MYRIAD': 'FP16',
    'MOVIDIUS': 'FP16',
}


def main():
    args = parse_args()
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)

    data_type = data_types[args.target]

    if not os.path.exists(args.training_dir):
        os.mkdir(args.training_dir)
    if args.all:
        if not args.facenet_graph:
            raise RuntimeError('Argument --facenet_graph is missing.')

        conver_onet(args.training_dir, data_type=data_type, prefix='onet.')
        convert_rnet(args.training_dir, data_type=data_type, prefix='rnet.')
        prepare_pnet(args.training_dir, data_type=data_type, prefix='pnet.')
        convert_facenet(
            args.training_dir, args.facenet_graph, data_type=data_type
        )
        if args.do_push:
            out_dir = os.path.join(args.training_dir, "openvino")
            push(None, out_dir)
        return
    if args.onet:
        conver_onet(args.training_dir, data_type=data_type, do_push=args.do_push)
    if args.rnet:
        convert_rnet(args.training_dir, data_type=data_type, do_push=args.do_push)
    if args.pnet:
        prepare_pnet(args.training_dir, data_type=data_type, do_push=args.do_push)
    if args.facenet:
        if not args.facenet_graph:
            raise RuntimeError('Argument --facenet_graph is missing.')

        convert_facenet(args.training_dir, args.facenet_graph, data_type=data_type, do_push=args.do_push)


if __name__ == "__main__":
    main()
