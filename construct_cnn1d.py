from tensorflow import keras as k


def construct_cnn1d(
        n_inputs: int,
        convolutional_layers: list[tuple[str, int]],
        fully_connected_layers: list[int],
        n_outputs: int,
        batchnorm_mode: str,
        dropout_mode: str,
        dropout_rate: float):
    """
    Constructs a 1D-CNN for multi-class classification based on layer size parameters.

    :param n_inputs:
            Specifies the number of input timesteps.
    :param convolutional_layers:
            Specifies the number and shapes of the convolutional layers.
            Each item in this list has the format (layer_type, n).
            layer_type may be "conv" or "pool"
            For layer_type "conv", n specifies the number of filters.
            For layer_type "pool", n specifies the pool size.
    :param fully_connected_layers:
            Specifies the number and sizes of the fully connnected layers.
            Fully connected layers come after all convolutional layers.
            The output layer should not be specified in this list.
    :param n_outputs:
            Specifies the number of output neurons/classes.
    :param batchnorm_mode:
            Specifies how batch normalization should be applied in the network.
            Batchnorm is applied after dropout when applicable.
            The following modes are supported:
            "all" inserts batch normalization prior to every convolutional and fully connected layer.
            "conv_only" inserts batch normalization prior to every convolutional layer.
            "fc_only" inserts batch normalization prior to every fully connected layer.
            "boundary_only" inserts batch normalization prior to the first fully connected layer.
            "none" turns batch normalization off.
    :param dropout_mode:
            Specifies how dropout should be applied in the network.
            Dropout is applied prior to batchnorm when applicable.
            The following modes are supported:
            "all" inserts dropout prior to every convolutional and fully connected layer.
            "conv_only" inserts dropout prior to every convolutional layer.
            "fc_only" inserts dropout prior to every fully connected layer.
            "boundary_only" inserts dropout prior to the first fully connected layer.
            "none" turns dropout off.
    :param dropout_rate:
            Specifies the dropout rate to use.
    :return:
            A model constructed according to input specfications.
    """

    if batchnorm_mode not in {"all", "conv_only", "fc_only", "boundary_only", "none"}:
        raise ValueError(f"[ERROR] Invalid batchnorm_mode {batchnorm_mode} specified.")
    if dropout_mode not in {"all", "conv_only", "fc_only", "boundary_only", "none"}:
        raise ValueError(f"[ERROR] Invalid dropout_mode {batchnorm_mode} specified.")

    inputs = k.Input(shape=(n_inputs, ), dtype=float)
    x = k.layers.Reshape((n_inputs, 1))(inputs)

    for layer_type, n in convolutional_layers:
        if layer_type == "conv":
            if dropout_mode in {"all", "conv_only"}:
                x = k.layers.Dropout(dropout_rate)(x)
            if batchnorm_mode in {"all", "conv_only"}:
                x = k.layers.BatchNormalization()(x)
            x = k.layers.Conv1D(filters=n, kernel_size=3, padding="same", activation="relu")(x)
        elif layer_type == "pool":
            x = k.layers.MaxPooling1D(pool_size=n)(x)
        else:
            raise ValueError(f"[ERROR] Invalid convolutional layer type {layer_type} specified.")

    x = k.layers.Flatten()(x)
    if dropout_mode in {"all", "fc_only", "boundary_only"}:
        x = k.layers.Dropout(dropout_rate)(x)
    if batchnorm_mode in {"all", "fc_only", "boundary_only"}:
        x = k.layers.BatchNormalization()(x)

    for n in fully_connected_layers:
        x = k.layers.Dense(units=n, activation="relu")(x)
        if dropout_mode in {"all", "fc_only"}:
            x = k.layers.Dropout(dropout_rate)(x)
        if batchnorm_mode in {"all", "fc_only"}:
            x = k.layers.BatchNormalization()(x)

    outputs = k.layers.Dense(units=n_outputs, activation="softmax")(x)

    model = k.Model(inputs, outputs)
    return model
