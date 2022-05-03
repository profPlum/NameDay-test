import numpy as np
import keras
import keras.layers as L

# this is an optimized diagonal of the dot product of a & b
np_diag_dot = lambda a, b: (a*b.T).sum(axis=1)
make_layer = lambda sz: L.Dense(sz)


def np_cos_similarity(x, y):
    """ x & y must be col vectors or matrices of column vectors! """
    y = y.reshape([y.shape[0], -1])
    x = x.reshape([x.shape[0], -1])
    return np_diag_dot(x.T, y) / (np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0))


def generate_nn_problem(layer_sizes, num_examples, add_noise=False):
    """
    generates a random problem a NN can solve (using a random neural network)
    :param layer_sizes: layer sizes of the upper bound of the size of the network needed to solve the problem
    :param num_examples: examples to create
    """
    # reversed because this is a generative model
    data_model = build_seq_model(reversed(layer_sizes))

    out_data = np.random.normal(size=(num_examples, layer_sizes[-1]))
    in_data = data_model.predict(out_data)
    # independent variables determine dependent variables

    if add_noise:
        noise = np.random.normal(size=in_data.shape) * 0.5
        in_data = in_data + noise

    return in_data, out_data


def build_seq_model(layers, dropout=0.0, input_first=True):
    """
    builds a simple sequential model + dropout
    either from layers or layer sizes (default is to use Input + Dense layers).
    """
    layers = list(layers)
    if type(layers[0]) is int:  # convert sizes to layers
        if input_first: layers = [L.InputLayer((layers[0],))] + [make_layer(sz) for sz in layers[1:]]
        else: layers = map(make_layer, layers)
    model = keras.models.Sequential()
    model.add(layers[0])  # input layer has neither dropout nor an activation
    for layer in layers[1:-1]:
        model.add(layer)
        model.add(L.Activation('tanh'))
        model.add(L.Dropout(rate=dropout))
    model.add(layers[-1])  # last layer has neither dropout nor an activation
    model.compile(keras.optimizers.Adam(), 'mse')
    model.build()
    return model


def invert_linear_layer(layer_output, layer=None, W=None, b=0):
    """ inverse a linear layer in keras using either parameters or the layer itself """
    return auto_name(layer_output, layer, W, b, forward=False).T  # auto name can invert a regular layer operation!
    # TODO: determine if transpose should be here or not?


def auto_name(known_embeddings, embedding_layer=None, W=None, b=0, forward=True):
    """
    derives new embeddings from known ones
    :param known_embeddings: embedding matrix (n_Hidden x PC), for either of the adjacent layers
    :param embedding_layer: the (keras) embedding layer to name the neurons of (or specify W & b)
    :param W: weights of embedding layer s.t. e=Wx + b
    :param b: biases of embedding layer s.t. e=Wx + b
    :param forward: whether we are propagating names forward or backward
    :return: returns embeddings as row vectors corresponding to embedding dims
    """
    if embedding_layer:
        weights = embedding_layer.get_weights()
        rotation_mat = weights[0].T
        biases = weights[1].reshape(-1, 1) if len(weights) > 1 else 0
    else:
        rotation_mat = W
        biases = b

    if forward: return rotation_mat.dot(known_embeddings) + biases
    else: return np.linalg.pinv(rotation_mat).dot(known_embeddings - biases)
    # verified mathematically that biases are handled properly
    # (NOTE: this assumes that we want to handle name embeddings in the same way we handle NN inputs)
