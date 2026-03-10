import os
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def _configure_tensorflow_gpu() -> None:
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
_configure_tensorflow_gpu()

def _board_shape(board_size: int) -> Tuple[int, int]:
    if board_size == 12:
        return 4, 3
    if board_size == 32:
        return 4, 8
    raise ValueError(f"Unsupported board size: {board_size}")

def build_q_model(
    board_size: int,
    action_size: int,
    learning_rate: float = 1e-3,
    embedding_dim: int = 32,
    num_channels: int = 96,
    num_residual_blocks: int = 4,
    aux_feature_dim: int = 1
) -> keras.Model:
    """
    DQN model: outputs Q-value for each action.
    Same backbone as policy-value model but with linear Q-value head.
    """
    board_h, board_w = _board_shape(board_size)
    if aux_feature_dim <= 0:
        raise ValueError("aux_feature_dim must be > 0")

    board_inputs = keras.Input(shape=(board_size,), dtype="int32", name="board")
    x = layers.Embedding(input_dim=16, output_dim=embedding_dim, name="chess_embedding")(board_inputs)
    x = layers.Reshape((board_h, board_w, embedding_dim), name="reshape_board")(x)
    aux_inputs = keras.Input(shape=(aux_feature_dim,), dtype="float32", name="aux")
    aux_embedding = layers.Dense(
        max(16, embedding_dim),
        activation="relu",
        name="aux_projection"
    )(aux_inputs)

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

    q = layers.Conv2D(2, kernel_size=1, padding="same", use_bias=False)(x)
    q = layers.BatchNormalization()(q)
    q = layers.Activation("relu")(q)
    q = layers.Flatten()(q)
    q = layers.Concatenate(name="q_head_with_aux")([q, aux_embedding])
    q = layers.Dense(256, activation="relu")(q)
    q = layers.Dense(action_size, activation="linear", name="q")(q)

    model = keras.Model(
        inputs={"board": board_inputs, "aux": aux_inputs},
        outputs=q,
        name="darkchess_q_net"
    )
    def q_loss(y_true, y_pred):
        """
        MSE summed over actions (not averaged).
        Since q_target matches q_pred for non-updated actions,
        only the updated action contributes to the loss.
        Using sum prevents gradient dilution by action_size.
        """
        return tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=q_loss,
    )
    return model