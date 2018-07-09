import align.detect_face as df
import tensorflow as tf
import os
import numpy as np
import subprocess

def conver_onet(dir):
    dir = os.path.join(dir,"onet")
    if not os.path.exists(dir):
        os.mkdir(dir)
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
        with tf.Session() as  sess:
            data = tf.placeholder(tf.float32, (1,48,48,3), 'input')
            with tf.variable_scope('onet'):
                onet = df.ONetMovidius({'data':data})
            with tf.variable_scope('onet',reuse=tf.AUTO_REUSE):
                onet.load(os.path.join('align', 'det3.npy'), sess)
            onet_output0 = graph.get_tensor_by_name('onet/conv6-1/conv6-1:0')
            onet_output1 = graph.get_tensor_by_name('onet/conv6-2/conv6-2:0')
            onet_output2 = graph.get_tensor_by_name('onet/conv6-3/conv6-3:0')
            onet_output01 = tf.reshape(onet_output0,[1,1,1,2])
            onet_output11 = tf.reshape(onet_output1,[1,1,1,4])
            onet_output21 = tf.reshape(onet_output2,[1,1,1,10])
            rnet_output = tf.concat([onet_output01, onet_output11,onet_output21],-1, name = 'onet/output0')
            tf.nn.max_pool(rnet_output, ksize = [1, 1, 1, 1], strides = [1, 1, 1, 1], padding = 'SAME',name='onet/output')
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(dir,'onet'))
            cmd = 'mvNCCompile movidius/onet/onet.meta -in input -on onet/output -o movidius/onet.graph'
            print(cmd)

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
        for f in tf.global_variables():
            print(f)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as  sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            with tf.variable_scope('rnet',reuse=tf.AUTO_REUSE):
                onet.load(os.path.join('align', 'det2.npy'), sess)

            saver.save(sess, os.path.join(dir,'rnet'))
            cmd = 'mvNCCompile movidius/rnet/rnet.meta -in input -on rnet/output -o movidius/rnet.graph'
            print(cmd)

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
        for f in tf.global_variables():
            print(f)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as  sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            with tf.variable_scope('pnet',reuse=tf.AUTO_REUSE):
                pnet.load(os.path.join('align', 'det1.npy'), sess)

            saver.save(sess, os.path.join(dir,'pnet'))
            cmd = 'mvNCCompile movidius/pnet/pnet.meta -in input -on pnet/output -o movidius/rnet.graph'
            print(cmd)


def preper_pnet(dir):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    factor_count=0
    h=300
    w=400
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
        name = '{}x{}'.format(hs,ws)
        print("-----------------------")
        conver_pnet(dir,name,hs,ws)
        cmd = 'mvNCCompile movidius/pnet/{}/pnet.meta -in input -on output -o movidius/pnet-{}.graph'.format(name,name)
        #out = subprocess.check_output(cmd, shell = True)
        print(cmd)
        print("-----------------------")
        break

def main():
    dir = 'movidius'
    if not os.path.exists(dir):
        os.mkdir(dir)
    #preper_pnet(dir)
    conver_onet(dir)
    conver_rnet(dir)
    conver_pnet(dir,28,38)

if __name__ == "__main__":
    main()