import tensorflow as tf
import models.inception_resnet_v1 as network
import os
from sys import argv
import sys

image_size=160
central_fraction = 0.875
model_base_file_name = ""

def print_usage():
    print("Usage: ")
    print("python3 convert_facenet.py model_base=<model_base_file_name>")
    print("    where <model_base_file_name> is the base file name of the saved tensorflow model files")
    print("    For example if your model files are: ")
    print("        facenet.index")
    print("        facenet.data-00000-of-00001")
    print("        facenet.meta")
    print("    then <model_base_file_name> is 'facenet' and you would pass model_base=facenet")


# handle the arguments
# return False if program should stop or True if args are ok
def handle_args():
    global model_base_file_name
    model_base_file_name = None
    for an_arg in argv:
        if (an_arg == argv[0]):
            continue

        elif (str(an_arg).lower() == 'help'):
            return False

        elif (str(an_arg).startswith('model_base=')):
            arg, val = str(an_arg).split('=', 1)
            model_base_file_name = str(val)
            print("model base file name is: " + model_base_file_name)
            return True

        else:
            return False

    if (model_base_file_name == None or len(model_base_file_name) < 1):
        return False

    return True



# This function is called from the entry point to do
# all the work.
def main():

    if (not handle_args()):
        # invalid arguments exit program
        print_usage()
        return 1

    with tf.Graph().as_default():
        image = tf.placeholder("float", shape=[1, image_size, image_size, 3], name='input')
        prelogits, _ = network.inference(image, 1.0, phase_train=False,bottleneck_layer_size=512)
        normalized = tf.nn.l2_normalize(prelogits, 1, name='l2_normalize')
        output = tf.identity(normalized, name='output')
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            print('Restoring ' + model_base_file_name)
            saver.restore(sess, model_base_file_name)

            # Save the network for fathom
            dir = os.path.dirname(model_base_file_name)
            dir = os.path.join(dir,'ncs')
            os.mkdir(dir)
            saver.save(sess,os.path.join(dir,'model'))


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())