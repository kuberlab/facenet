import align.detect_face as df
import tensorflow as tf
import os
import numpy as np
import subprocess
import logging
import argparse
from movidius_tools.tools import parse_check_ouput
import datetime


def submit(params):
    if os.environ.get('PROJECT_ID',None) is not None:
        from mlboardclient.api import client
        client.update_task_info(params)

def push(model,dirame):
    if os.environ.get('PROJECT_ID',None) is not None:
        from mlboardclient.api import client
        timestamp = datetime.datetime.now().strftime('%s')
        version = '1.0.0-%s' % timestamp
        mlboard = client.Client()
        mlboard.model_upload(model, version, dirame)
        submit({'model':'{}:{}'.format(model,version)})
        logging.info("New model uploaded as '%s', version '%s'." % (model, version))

def conver_onet(dir,prefix=None,do_push=False):
    out_dir = os.path.join(dir,"movidius")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dir = os.path.join(dir,"onet")
    if not os.path.exists(dir):
        os.mkdir(dir)

    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        with tf.Session() as  sess:
            data = tf.placeholder(tf.float32, (1,48,48,3), 'input')
            logging.info("Load ONET graph")
            with tf.variable_scope('onet'):
                onet = df.ONetMovidius({'data':data})
            logging.info("Load ONET weights")
            with tf.variable_scope('onet',reuse=tf.AUTO_REUSE):
                onet.load(os.path.join('align', 'det3.npy'), sess)
            logging.info("Create ONET ouput layer")
            onet_output0 = graph.get_tensor_by_name('onet/conv6-1/conv6-1:0')
            onet_output1 = graph.get_tensor_by_name('onet/conv6-2/conv6-2:0')
            onet_output2 = graph.get_tensor_by_name('onet/conv6-3/conv6-3:0')
            onet_output01 = tf.reshape(onet_output0,[1,1,1,2])
            onet_output11 = tf.reshape(onet_output1,[1,1,1,4])
            onet_output21 = tf.reshape(onet_output2,[1,1,1,10])
            rnet_output = tf.concat([onet_output01, onet_output11,onet_output21],-1, name = 'onet/output0')
            tf.nn.max_pool(rnet_output, ksize = [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME',name='onet/output')
            saver = tf.train.Saver()
            logging.info("Freeze ONET graph")
            saver.save(sess, os.path.join(dir,'onet'))
            cmd = 'mvNCCheck {}/onet.meta -in input -on onet/output -s 12 -cs 0,1,2 -S 255'.format(dir)
            logging.info('Validate Movidius: %s',cmd)
            result = subprocess.check_output(cmd, shell=True).decode()
            logging.info(result)
            result = parse_check_ouput(result,prefix=prefix)
            submit(result)
            cmd = 'mvNCCompile {}/onet.meta -in input -on onet/output -o {}/onet.graph -s 12'.format(dir,out_dir)
            logging.info('Compile: %s',cmd)
            result = subprocess.check_output(cmd, shell=True).decode()
            logging.info(result)
            if do_push:
                push('facenet-onet',out_dir)

def conver_rnet(dir,prefix=None,do_push=False):
    out_dir = os.path.join(dir,"movidius")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dir = os.path.join(dir,"rnet")
    if not os.path.exists(dir):
        os.mkdir(dir)
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        logging.info("Load RNET graph")
        data = tf.placeholder(tf.float32, (1,24,24,3), 'input')
        with tf.variable_scope('rnet'):
            onet = df.RNetMovidius({'data':data})
        logging.info("Create RNET output")
        rnet_output0 = graph.get_tensor_by_name('rnet/conv5-1/conv5-1:0')
        rnet_output1 = graph.get_tensor_by_name('rnet/conv5-2/conv5-2:0')
        rnet_output2 = tf.reshape(rnet_output0,[1,1,1,2])
        rnet_output3 = tf.reshape(rnet_output1,[1,1,1,4])
        rnet_output = tf.concat([rnet_output2, rnet_output3],-1, name = 'rnet/output0')
        tf.nn.max_pool(rnet_output, ksize = [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME',name='rnet/output')
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as  sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            logging.info("Load RNET weights")
            with tf.variable_scope('rnet',reuse=tf.AUTO_REUSE):
                onet.load(os.path.join('align', 'det2.npy'), sess)
            logging.info("Freeze RNET graph")
            saver.save(sess, os.path.join(dir,'rnet'))

            cmd = 'mvNCCheck {}/rnet.meta -in input -on rnet/output -s 12 -cs 0,1,2 -S 255'.format(dir)
            logging.info('Validate Movidius: %s',cmd)
            result = subprocess.check_output(cmd, shell=True).decode()
            logging.info(result)
            result = parse_check_ouput(result,prefix=prefix)
            submit(result)
            cmd = 'mvNCCompile {}/rnet.meta -in input -on rnet/output -o {}/rnet.graph -s 12'.format(dir,out_dir)
            logging.info('Compile: %s',cmd)
            result = subprocess.check_output(cmd, shell=True).decode()
            logging.info(result)
            if do_push:
                push('facenet-rnet',out_dir)

def conver_pnet(dir,h,w):
    logging.info("Prepare PNET-{}x{} graph".format(h,w))
    out_dir = os.path.join(dir,"movidius")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dir = os.path.join(dir,'pnet-{}x{}'.format(h,w))
    if not os.path.exists(dir):
        os.mkdir(dir)
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        logging.info("Load PNET graph")
        data = tf.placeholder(tf.float32, (1,w,h,3), 'input')
        with tf.variable_scope('pnet'):
            pnet = df.PNetMovidius({'data':data})
        logging.info("Create PNET output")
        pnet_output0 = graph.get_tensor_by_name('pnet/conv4-1/BiasAdd:0')
        pnet_output1 = graph.get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
        pnet_output = tf.concat([pnet_output0, pnet_output1],-1, name = 'pnet/output0')
        tf.nn.max_pool(pnet_output, ksize = [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME',name='pnet/output')
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as  sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            logging.info("Load PNET weights")
            with tf.variable_scope('pnet',reuse=tf.AUTO_REUSE):
                pnet.load(os.path.join('align', 'det1.npy'), sess)

            logging.info("Freeze PNET graph")
            saver.save(sess, os.path.join(dir,'pnet'))

            cmd = 'mvNCCheck {}/pnet.meta -in input -on pnet/output -s 12 -cs 0,1,2 -S 255'.format(dir)
            logging.info('Validate Movidius: %s',cmd)
            result = subprocess.check_output(cmd, shell=True).decode()
            logging.info(result)
            result = parse_check_ouput(result,'{}x{}'.format(h,w))
            submit(result)
            cmd = 'mvNCCompile {}/pnet.meta -in input -on pnet/output -o {}/pnet-{}x{}.graph -s 12'.format(dir,out_dir,h,w)
            logging.info('Compile: %s',cmd)
            result = subprocess.check_output(cmd, shell=True).decode()
            logging.info(result)


def prepare_pnet(dir,do_push=False):
    minsize = 20  # minimum size of face
    factor = 0.709  # scale factor
    factor_count=0
    h=480
    w=680
    minl=np.amin([h, w])
    m=12.0/minsize
    minl=minl*m
    scales=[]
    while minl>=12:
        scales += [m*np.power(factor, factor_count)]
        minl = minl*factor
        factor_count += 1
    # first stage
    for scale in scales:
        hs=int(np.ceil(h*scale))
        ws=int(np.ceil(w*scale))
        conver_pnet(dir,hs,ws)
    if do_push:
        out_dir = os.path.join(dir,"movidius")
        push('facenet-pnet',out_dir)

def convert_facenet(dir,model_base_path,image_size,output_size,prefix=None,do_push=False):
    import facenet
    from models import inception_resnet_v1
    out_dir = os.path.join(dir,"movidius")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    dir = os.path.join(dir,"facenet")
    if not os.path.exists(dir):
        os.mkdir(dir)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        logging.info("Load FACENET graph")
        image = tf.placeholder("float", shape=[1, image_size, image_size, 3], name='input')
        prelogits, _ = inception_resnet_v1.inference(image, 1.0, phase_train=False, bottleneck_layer_size=output_size)
        normalized = tf.nn.l2_normalize(prelogits, 1, name='l2_normalize')
        output = tf.identity(normalized, name='output')

        # Do not remove
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            base_name = model_base_path
            meta_file, ckpt_file = facenet.get_model_filenames(base_name)
            logging.info("Restore FACENET graph from %s %s",meta_file, ckpt_file)
            saver = tf.train.import_meta_graph(os.path.join(base_name, meta_file))
            saver.restore(sess, os.path.join(base_name, ckpt_file))

            logging.info("Freeze FACENET graph")
            saver.save(sess, os.path.join(dir,'facenet'))

            cmd = 'mvNCCheck {}/facenet.meta -in input -on output -s 12 -cs 0,1,2 -S 255'.format(dir)
            logging.info('Validate Movidius: %s',cmd)
            result = subprocess.check_output(cmd, shell=True).decode()
            logging.info(result)
            result = parse_check_ouput(result,prefix=prefix)
            submit(result)
            cmd = 'mvNCCompile {}/facenet.meta -in input -on output -o {}/facenet.graph -s 12'.format(dir,out_dir)
            logging.info('Compile: %s',cmd)
            result = subprocess.check_output(cmd, shell=True).decode()
            logging.info(result)
            if do_push:
                push('movidius-facenet',out_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--training_dir',
        help='Training dir'
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
        help='Build RNET'
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
        'model_base_path',
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

def main():
    args = parse_args()
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    if not os.path.exists(args.training_dir):
        os.mkdir(args.training_dir)
    if args.all:
        conver_onet(args.training_dir,prefix='onet')
        conver_rnet(args.training_dir,prefix='rnet')
        prepare_pnet(args.training_dir)
        convert_facenet(dir,args.model_base_path,args.image_size,args.output_size,prefix='facenet',do_push=True)
        return
    if args.onet:
        conver_onet(args.training_dir,do_push=args.do_push)
    if args.rnet:
        conver_rnet(args.training_dir,do_push=args.do_push)
    if args.pnet:
        prepare_pnet(args.training_dir,do_push=args.do_push)
    if args.rnet:
        convert_facenet(dir,args.model_base_path,args.image_size,args.output_size,do_push=True)



if __name__ == "__main__":
    main()