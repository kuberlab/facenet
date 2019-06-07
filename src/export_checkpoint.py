import argparse
from os import path

import tensorflow as tf
from tensorflow.python.framework import graph_util

import facenet


def freeze_graph(model_dir, output_nodes='embeddings',
                 output_filename='facenet.pb',
                 rename_outputs=None):
    # Load checkpoint
    meta_file, ckpt_file = facenet.get_model_filenames(model_dir)
    meta_file = path.join(model_dir, meta_file)
    ckpt_file = path.join(model_dir, ckpt_file)

    output_graph = output_filename

    print('Importing meta graph...')
    # Devices should be cleared to allow Tensorflow to control placement of
    # graph when loading on different machines
    saver = tf.train.import_meta_graph(
        meta_file, clear_devices=True
    )

    graph = tf.get_default_graph()

    onames = output_nodes.split(',')

    # https://stackoverflow.com/a/34399966/4190475
    if rename_outputs is not None:
        nnames = rename_outputs.split(',')
        with graph.as_default():
            for o, n in zip(onames, nnames):
                _out = tf.identity(graph.get_tensor_by_name(o + ':0'), name=n)
            onames = nnames

    input_graph_def = graph.as_graph_def()

    # fix batch norm nodes
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, ckpt_file)

        # In production, graph weights no longer need to be updated
        # graph_util provides utility to change all variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def,
            onames  # unrelated nodes will be discarded
        )

        # Serialize and write to file
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        print("Saved to %s." % output_graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prune and freeze weights from checkpoints into production models')
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint files",
    )
    parser.add_argument(
        "--output_nodes",
        default='embeddings',
        type=str,
        help="Names of output node, comma seperated"
    )
    parser.add_argument(
        "--output",
        default='facenet.pb',
        type=str,
        help="Output graph filename"
    )
    parser.add_argument(
        "--rename_outputs",
        default=None,
        type=str,
        help=(
            "Rename output nodes for better"
            "readability in production graph, to be specified in"
            "the same order as output_nodes"
        )
    )
    args = parser.parse_args()

    freeze_graph(args.checkpoint_path, args.output_nodes, args.output, args.rename_outputs)
