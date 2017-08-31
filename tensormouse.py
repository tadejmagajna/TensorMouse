import argparse
import os
import sys
import json
from tensormouse import workers

if __name__ == '__main__':

    #path to tenforblow object detection graph
    path_to_graph = os.path.join(os.getcwd(), 'graphs', 'ssd_mobilenet_v1_coco_11_06_2017', 'frozen_inference_graph.pb')
    path_to_labels = os.path.join(os.getcwd(), 'graphs', 'labels.json')

    #set default arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera. (default: 0)')
    parser.add_argument('-obj', '--object', dest='object_name', type=str,
                        default='cup', help='COCO dataset object name (default: cup)')
    parser.add_argument('-graphpath', '--graphpath', dest='graph_path', type=str,
                        default=path_to_graph, help='Path to tensofrlow object detection frozen graph (default: frozen_inference_graph.pb)')
    args = parser.parse_args()

    #get object_id if exists in labels json file
    with open(path_to_labels) as labels_json:
        labels = json.load(labels_json)
    if args.object_name in labels.keys():
        object_id = labels[args.object_name]
    else:
        print("Object '" + args.object_name + "' not found. Please refer to " + path_to_labels + ' to see the full list of available objects')
        sys.exit()
    
    #start main worker
    workers.main_worker(object_id, args.video_source, args.object_name)
