
# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
import tensorflow as tf
import argparse 
import numpy as np

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
        return graph

def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--model",  
     default="frozen_model.pb", type=str, help="Frozen model file to import")
     args = parser.parse_args()
     loaded_graph = tf.Graph()
     with tf.Session(graph=loaded_graph) as sess:
         tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.model)
         y = loaded_graph.get_tensor_by_name("input:0")
         

main()