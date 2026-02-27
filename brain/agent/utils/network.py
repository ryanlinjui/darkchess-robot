from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers

def _board_shape(board_size: int) -> Tuple[int, int]:
    if board_size == 12:
        return 4, 3
    if board_size == 32:
        return 4, 8
    raise ValueError(f"Unsupported board size: {board_size}")

def build_policy_value_model(
    board_size: int,
    action_size: int,
    learning_rate: float,
    embedding_dim: int = 32,
    num_channels: int = 96,
    num_residual_blocks: int = 4,
    value_hidden_units: int = 128,
) -> keras.Model:
    """
    AlphaZero-style policy-value model for darkchess.
    Input is a 1D encoded board; outputs are:
    - pi: action distribution over the full action space
    - v: state value in [-1, 1]
    """
    board_h, board_w = _board_shape(board_size)

    inputs = keras.Input(shape=(board_size,), dtype="int32", name="board")
    x = layers.Embedding(input_dim=16, output_dim=embedding_dim, name="chess_embedding")(inputs)
    x = layers.Reshape((board_h, board_w, embedding_dim), name="reshape_board")(x)

    x = layers.Conv2D(num_channels, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for idx in range(num_residual_blocks):
        residual = x

        y = layers.Conv2D(num_channels, kernel_size=3, padding="same", use_bias=False)(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)
        y = layers.Conv2D(num_channels, kernel_size=3, padding="same", use_bias=False)(y)
        y = layers.BatchNormalization()(y)

        x = layers.Add(name=f"residual_add_{idx}")([residual, y])
        x = layers.Activation("relu")(x)

    p = layers.Conv2D(2, kernel_size=1, padding="same", use_bias=False)(x)
    p = layers.BatchNormalization()(p)
    p = layers.Activation("relu")(p)
    p = layers.Flatten()(p)
    pi = layers.Dense(action_size, activation="softmax", name="pi")(p)

    v = layers.Conv2D(1, kernel_size=1, padding="same", use_bias=False)(x)
    v = layers.BatchNormalization()(v)
    v = layers.Activation("relu")(v)
    v = layers.Flatten()(v)
    v = layers.Dense(value_hidden_units, activation="relu")(v)
    v = layers.Dense(1, activation="tanh", name="v")(v)

    model = keras.Model(inputs=inputs, outputs=[pi, v], name="darkchess_policy_value_net")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"pi": "categorical_crossentropy", "v": "mean_squared_error"},
        loss_weights={"pi": 1.0, "v": 1.0},
    )
    return model