import os
from keras.models import model_from_json
import tensorflow as tf

def init():
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_json = os.path.join(THIS_FOLDER, 'model.json')
    my_model = os.path.join(THIS_FOLDER, 'model.h5')

    json_file = open(my_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # cargamos los pesos de nuestro modelo
    loaded_model.load_weights(my_model)
    print("Loaded Model from disk")

    # compilamos y evaluamos el modelo
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    graph = tf.get_default_graph()

    return loaded_model, graph
