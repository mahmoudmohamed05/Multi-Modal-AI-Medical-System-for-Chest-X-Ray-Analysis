
"""DenseNet121 + Cross-Attention model architecture."""
from functools import lru_cache

IMG_SIZE = (224, 224)
INPUT_SHAPE = (*IMG_SIZE, 3)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MODEL_NAME = "DenseNet121 + Cross-Attention"


@lru_cache(maxsize=1)
def _positional_embedding_cls():
    import keras
    from keras import layers

    @keras.utils.register_keras_serializable(package="xray")
    class LearnedPositionalEmbedding(layers.Layer):
        def build(self, input_shape):
            self.pos = self.add_weight(
                name="pos",
                shape=(1, input_shape[1], input_shape[2]),
                initializer=keras.initializers.RandomNormal(stddev=0.02),
                trainable=True,
            )
            super().build(input_shape)

        def call(self, x):
            return x + self.pos

    return LearnedPositionalEmbedding


def build_model(
    d_model=256,
    num_heads=8,
    dropout=0.1,
    imagenet_weights=False,
):
    import keras
    from keras import layers

    LearnedPositionalEmbedding = _positional_embedding_cls()

    base = keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet" if imagenet_weights else None,
        input_shape=INPUT_SHAPE,
    )

    extractor = keras.Model(
        base.input,
        [
            base.get_layer("conv4_block24_concat").output,
            base.output,
        ],
        name="densenet_features",
    )
    extractor.trainable = False

    inputs = keras.Input(INPUT_SHAPE, name="image")
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Keep the training augmentation layers in the architecture.
    # Keras disables random augmentation when model.predict() is used.
    x = keras.Sequential(
        [
            layers.RandomRotation(0.03),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )(x)

    x = layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2],
        name="imagenet_norm",
    )(x)

    mid, deep = extractor(x, training=False)

    kv = layers.Reshape((-1, mid.shape[-1]))(mid)
    q = layers.Reshape((-1, deep.shape[-1]))(deep)

    kv = LearnedPositionalEmbedding(name="pos_kv")(
        layers.Dense(d_model, name="proj_kv")(kv)
    )
    q = LearnedPositionalEmbedding(name="pos_q")(
        layers.Dense(d_model, name="proj_q")(q)
    )

    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout,
        name="cross_attention",
    )(query=q, value=kv, key=kv)

    x = layers.LayerNormalization(epsilon=1e-6)(
        q + layers.Dropout(dropout)(attn)
    )

    ffn = layers.Dense(d_model * 2, activation="gelu")(x)
    ffn = layers.Dropout(dropout)(ffn)
    ffn = layers.Dense(d_model)(ffn)

    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        1, activation="sigmoid", name="pneumonia_prob"
    )(x)

    return keras.Model(
        inputs, outputs, name="densenet121_cross_attention"
    )
