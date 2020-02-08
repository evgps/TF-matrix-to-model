import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Add
tf.compat.v1.disable_eager_execution()


def add_InputLayer(cfg, inputs):
    shape = [cfg['input_shape'][0][1], cfg['input_shape'][0][2], cfg['input_shape'][0][3]]
    return Input(shape, 1)

def add_Dense(cfg, inputs):
    return Dense(cfg['units'], cfg['activation'])(inputs[0])

def add_Conv2D(cfg, inputs):
    return Conv2D(cfg['filters'], cfg['kernel_size'], cfg['strides'], cfg['padding'])(inputs[0])

def add_Add(cfg, inputs):
    return Add()(inputs)

def add_MaxPooling2D(cfg, inputs):
    return MaxPooling2D(cfg['pool_size'], cfg['strides'], cfg['padding'])(inputs[0])

def add_GlobalAveragePooling2D(cfg, inputs):
    return GlobalAveragePooling2D(cfg['data_format'])(inputs[0])

def add_ZeroPadding2D(cfg, inputs):
    return ZeroPadding2D(cfg['padding'])(inputs[0])

def add_Activation(cfg, inputs):
    return Activation(cfg['activation'])(inputs[0])

def add_BatchNormalization(cfg, inputs):
    return BatchNormalization(cfg['axis'], cfg['momentum'], cfg['epsilon'])(inputs[0])

add_func_dict = {'InputLayer' : add_InputLayer,
                'Dense' : add_Dense,
                'ZeroPadding2D' : add_ZeroPadding2D,
                'Activation' : add_Activation,
                'BatchNormalization'   : add_BatchNormalization,
                'MaxPooling2D' : add_MaxPooling2D,
                'Conv2D' : add_Conv2D,
                'Add'  : add_Add,
                'GlobalAveragePooling2D' : add_GlobalAveragePooling2D}

# data format: [layer_classname, cfg {'units' : 1, 'activation' : None}]
def get_type(layer):
    return str(layer.__class__.__name__)

def parse_layer(layer):
    attr_dict = {'InputLayer' : ['input_shape', 'batch_size'],
                'Dense' : ['units', 'activation'],
                'ZeroPadding2D' : ['padding'],
                'Activation' : ['activation'],
                'BatchNormalization'   : ['axis', 'momentum', 'epsilon'],
                'MaxPooling2D' : ['pool_size', 'strides', 'padding'],
                'Conv2D' : ['filters', 'kernel_size', 'strides', 'padding'],
                'Add'  : [],
                'GlobalAveragePooling2D' : ['data_format']}
    cfg = dict()
    for attr in attr_dict[get_type(layer)]:
        cfg[attr] = getattr(layer, attr)
    return cfg

class Node(object):
    def __init__(self, cfg, layer_type):
        self.inputs = []
        self.cfg = cfg
        self.layer_type = layer_type
        self.out = None

def out_from(node):
    inputs = []
    if node.out is not None:
        return node.out
    for parent in node.inputs:
        inputs.append(out_from(parent))
    node.out = add_func_dict[node.layer_type](node.cfg, inputs)
    return node.out
    