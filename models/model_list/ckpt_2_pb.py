import tensorflow as tf
import os
from tensorflow.python.framework.graph_util import convert_variables_to_constants
path = os.getcwd()



output_path = 'ckpt_converted_pb_models/model.pb'
with tf.Session() as sess:
    graph = tf.get_default_graph()

    saver = tf.train.import_meta_graph('models/checkpoints0/model.meta', graph=graph)
    saver.restore(sess, tf.train.latest_checkpoint('models/checkpoints0'))
    for op in graph.get_operations():
        print(op.name)
    output_node_names = ['sigmoid_out']
    output_graph_def = convert_variables_to_constants(sess=sess, input_graph_def=sess.graph_def, output_node_names=output_node_names)
    with tf.gfile.FastGFile(output_path, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

