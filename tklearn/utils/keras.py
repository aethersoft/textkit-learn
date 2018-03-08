from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D


def global_pooling_1d(t):
    """
    Creates a 1D Global Pooling Layer

    :param t: type of the pooling layer
    :return: Max/Avg Pooling layer as specified by type param
    """
    if t == 'max':
        return GlobalMaxPooling1D()
    elif t == 'avg':
        return GlobalAveragePooling1D()
    else:
        assert False, 'Unidentified pooling method named {}. Please use one of {}'.format(t, ['max', 'avg'])
