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
            cmd = 'mvNCCompile movidius/onet/onet.meta -in input -on onet/proxy -o movidius/onet.graph'
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
            cmd = 'mvNCCompile movidius/rnet/rnet.meta -in input -on rnet/proxy -o movidius/rnet.graph'
            print(cmd)

def conver_pnet(dir,scale,h,w):
    dir = os.path.join(dir,"pnet",scale)
    if not os.path.exists(dir):
        os.mkdir(dir)
    tf.reset_default_graph()
    with tf.Session() as  sess:
        data = tf.placeholder(tf.float32, (1,h,w,3), 'input')
        with tf.variable_scope('pnet'):
            pnet = df.PNetMovidius({'data':data})
        with tf.variable_scope('pnet',reuse=tf.AUTO_REUSE):
            pnet.load(os.path.join('align', 'det1.npy'), sess)
        o1 = tf.get_default_graph().get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
        o2 = tf.get_default_graph().get_tensor_by_name('pnet/conv4-1/BiasAdd:0')
        #o1 = tf.reshape(o1,[1,1,1,(int(h/2)-5)*(int(w/2)-5)*4])
        #o2 = tf.reshape(o2,[1,1,1,(int(h/2)-5)*(int(w/2)-5)*2])
        #o3 = tf.pad(o1, [[0, 0],[0, 0], [0, 0], [0, 2]])
        #o1 = tf.pad(o1, [[0,1]])
        #o2 = tf.pad(o2, [[1,0]])
        #o = tf.concat([o1,o2],axis=3,name='proxy')
        #proxy= tf.nn.max_pool(o,ksize=[1, 1, 1, 1],strides=[1, 1, 1, 1], padding='SAME', name='proxy')
        #o = tf.concat([o1,o2], 3)
        #o = tf.multiply(o,1,name='output')
        #tf.identity(o,name='output')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir,'pnet'))


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

if __name__ == "__main__":
    main()