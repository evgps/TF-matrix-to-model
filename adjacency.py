import os
import tensorflow as tf
import numpy as np

def hash_fuction(l):
    '''calc hash for layer l'''
    return hash((l.__class__, tuple(l.input_shape), tuple(l.output_shape), tuple(tuple(x.get_shape()) for x in l.weights),l.count_params(), l.dtype))


def to_adjacency_matrix(model):
    '''function to convert model -> model_layers_db [num, name, hash] and adjactncy matrix '''
    dot = tf.keras.utils.model_to_dot(
        model,
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96,
        subgraph=True
    )
    layer_indexes = {}
    model_layers = []
    for i, layer in enumerate(model.layers):
        model_layers.append([i, layer.name, layer, hash_fuction(layer)])
        layer_indexes[layer.name] = i

    layers = {}
    for node in dot.get_node_list():
        layer = node.to_string().split('label="')[1].split(":")[0]
        layers[node.to_string().split(' ')[0]] = layer#model.get_layer(layer)

    keylist = list(set(layers.keys()))
    matrix = np.zeros((len(keylist),len(keylist)))
    for edge in dot.get_edge_list():
        ll = edge.to_string().replace(';','').split(' -- ')
        i = layer_indexes[layers[ll[0]]]
        j = layer_indexes[layers[ll[1]]]
        matrix[i][j] = 1
        matrix[j][i] = 1

    return matrix, model_layers