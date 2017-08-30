import tensorflow as tf
import os

PATH_TO_GRAPH = os.path.join(os.getcwd(), 'graphs', 'ssd_mobilenet_v1_coco_11_06_2017', 'frozen_inference_graph.pb')

def graph_and_sess():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
        sess = tf.Session(graph=detection_graph)

    return detection_graph, sess

