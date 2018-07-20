import align.detect_face as df
import tensorflow as tf
import os
import numpy as np
import subprocess
import logging
import argparse
from movidius_tools.tools import parse_check_ouput



def conver_onet(dir):
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
            logging.info(parse_check_ouput(result))
            cmd = 'mvNCCompile {}/onet.meta -in input -on onet/output -o {}/onet.graph -s 12'.format(dir,out_dir)
            logging.info('Compile: %s',cmd)
            result = subprocess.check_output(cmd, shell=True).decode()
            logging.info(result)

def conver_rnet(dir):
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        dir = os.path.join(dir,"rnet")
        if not os.path.exists(dir):
            os.mkdir(dir)
        data = tf.placeholder(tf.float32, (1,24,24,3), 'input')
        with tf.variable_scope('rnet'):
            onet = df.RNetMovidius({'data':data})

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
            with tf.variable_scope('rnet',reuse=tf.AUTO_REUSE):
                onet.load(os.path.join('align', 'det2.npy'), sess)

            saver.save(sess, os.path.join(dir,'rnet'))
            cmd = 'mvNCCompile movidius/rnet/rnet.meta -in input -on rnet/output -o movidius/rnet.graph -s 12'
            print(cmd)
            subprocess.call(cmd, shell=True)

def conver_pnet(dir,h,w):
    dir = os.path.join(dir,'pnet-{}x{}'.format(h,w))
    if not os.path.exists(dir):
        os.mkdir(dir)
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        data = tf.placeholder(tf.float32, (1,w,h,3), 'input')
        with tf.variable_scope('pnet'):
            pnet = df.PNetMovidius({'data':data})
        pnet_output0 = graph.get_tensor_by_name('pnet/conv4-1/BiasAdd:0')
        pnet_output1 = graph.get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
        pnet_output = tf.concat([pnet_output0, pnet_output1],-1, name = 'pnet/output0')
        tf.nn.max_pool(pnet_output, ksize = [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME',name='pnet/output')
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as  sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            with tf.variable_scope('pnet',reuse=tf.AUTO_REUSE):
                pnet.load(os.path.join('align', 'det1.npy'), sess)

            saver.save(sess, os.path.join(dir,'pnet'))
            cmd = 'mvNCCompile {}/pnet.meta -in input -on pnet/output -o movidius/pnet-{}x{}.graph -s 12'.format(dir,h,w)
            print(cmd)
            subprocess.call(cmd, shell=True)



def preper_pnet(dir):
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
    return parser.parse_args()

def main():
    args = parse_args()
    logging.getLogger().setLevel('INFO')
    tf.logging.set_verbosity(tf.logging.INFO)
    if not os.path.exists(args.training_dir):
        os.mkdir(args.training_dir)
    if args.onet:
        conver_onet(args.training_dir)

def main1():
    dir = 'movidius'
    if not os.path.exists(dir):
        os.mkdir(dir)
    #preper_pnet(dir)
    conver_onet(dir)
    conver_rnet(dir)
    preper_pnet(dir)

if __name__ == "__main__":
    main()