import argparse
import os
import shutil
import subprocess
import sys

import tensorflow as tf

import facenet
from models import inception_resnet_v1

image_size=160
central_fraction = 0.875
tf.logging.set_verbosity(tf.logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_base_path',
        help='Base model path'
    )
    parser.add_argument(
        '--output_file',
        '-O',
        default='facenet.graph'
    )
    parser.add_argument(
        '--output-size',
        type=int,
        default=512,
        help='Base model path'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help=(
            'Check graph on movidius stick before compile. '
            'Note: it requires working movidius stick.'
        )
    )



    return parser.parse_args()


# This function is called from the entry point to do
# all the work.
def main():
    args = parse_args()
    out_dir = '/tmp/facenet'

    with tf.Graph().as_default():
        image = tf.placeholder("float", shape=[1, image_size, image_size, 3], name='input')
        prelogits, _ = inception_resnet_v1.inference(image, 1.0, phase_train=False, bottleneck_layer_size=args.output_size)
        normalized = tf.nn.l2_normalize(prelogits, 1, name='l2_normalize')
        output = tf.identity(normalized, name='output')

        # Do not remove
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            base_name = args.model_base_path
            meta_file, ckpt_file = facenet.get_model_filenames(base_name)
            saver = tf.train.import_meta_graph(os.path.join(base_name, meta_file))
            saver.restore(sess, os.path.join(base_name, ckpt_file))

            # Save the network for fathom
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)

            saver.save(sess, out_dir + '/facenet')

    if args.check:
        cmd = 'mvNCCheck {0}/facenet.meta -in input -on output -s 12'.format(out_dir)
        print('Running check:\n')
        print(cmd)
        print('')
        print(subprocess.check_output(cmd, shell=True).decode())

    cmd = 'mvNCCompile {0}/facenet.meta -in input -on output -o {1} -s 12'.format(out_dir, args.output_file)

    print('Run:\n')
    print(cmd)
    print('')
    print(subprocess.check_output(cmd, shell=True).decode())

    shutil.rmtree(out_dir)


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
