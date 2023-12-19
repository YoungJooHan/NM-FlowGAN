
import os
from importlib import import_module

flow_layer_class_dict = {}

def regist_layer(layer_class):
    layer_name = layer_class.__name__.lower()
    assert not layer_name in flow_layer_class_dict, 'there is already registered layer: %s in flow_layer_class_dict.' % layer_name
    flow_layer_class_dict[layer_name] = layer_class
    return layer_class

def get_flow_layer(layer_name:str):
    layer_name = layer_name.lower()
    return flow_layer_class_dict[layer_name]

# import all python files in model folder
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    import_module('core.model.flow_layers.{}'.format(module[:-3]))
del module