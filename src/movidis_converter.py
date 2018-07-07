import align.detect_face as df
import tensorflow as tf
import os
import numpy as np
import subprocess

def conver_pnet(dir,scale,h,w):
    dir = os.path.join(dir,"pnet",scale)
    if not os.path.exists(dir):
        os.mkdir(dir)
    tf.reset_default_graph()
    with tf.Session() as  sess:
        data = tf.placeholder(tf.float32, (1,h,w,3), 'input')
        with tf.variable_scope('pnet'):
            pnet = df.PNet({'data':data})
        with tf.variable_scope('pnet',reuse=tf.AUTO_REUSE):
            pnet.load(os.path.join('align', 'det1.npy'), sess)
        o1 = tf.get_default_graph().get_tensor_by_name('pnet/conv4-2/BiasAdd:0')
        o2 = tf.get_default_graph().get_tensor_by_name('pnet/fake_prob:0')
        o1 = tf.pad(o1, [[0, 0],[0, 0], [0, 0], [0, 2]])
        o2 = tf.pad(o2, [[0, 0],[0, 0], [0, 0], [4, 0]])
        o = tf.add(o1,o2)
        #proxy= tf.nn.max_pool(o2,ksize=[1, 1, 1, 1],strides=[1, 1, 1, 1], padding='SAME', name='proxy')
        #o = tf.concat([o1,o2], 3)
        #o = tf.multiply(o,1,name='output')
        tf.identity(o1,name='output')
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
    preper_pnet(dir)

if __name__ == "__main__":
    main()