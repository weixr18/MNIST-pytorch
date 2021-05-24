import os

WS_PATH = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "."
).replace('\\', '/') + '/'
MODEL_PATH = os.path.abspath(WS_PATH + '../saved/').replace('\\', '/') + '/'
