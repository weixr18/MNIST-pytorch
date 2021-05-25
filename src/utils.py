import os

WS_PATH = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "."
).replace('\\', '/') + '/'
MODEL_PATH = os.path.abspath(WS_PATH + '../saved/').replace('\\', '/') + '/'
DATA_PATH = os.path.abspath(WS_PATH + '../data/').replace('\\', '/') + '/'


def model_class(model_type):
    if(model_type == 'mlpmixer' or model_type == 'vfnet'):
        return 'fancy'
    elif(model_type[:3] == 'mlp'):
        return 'mlp'
    elif(model_type == 'lenet'):
        return 'cnn'
    elif(model_type == 'vit'):
        return 'attention'
    else:
        print("Utils: model {0} not supported.".format(model_type))
        return None
