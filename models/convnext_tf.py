import tensorflow as tf
from tensorflow.keras import layers
from typing import List


class ConvNormAct(layers.Layer):
    """
    A little util layer composed by (conv) -> (norm) -> (act) layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        norm=layers.BatchNormalization,
        act=layers.ReLU,
        **kwargs
    ):
        super().__init__()
        self.conv = layers.Conv2D(
            out_features,
            kernel_size=kernel_size,
            padding="same",
            **kwargs,
        )
        self.norm = norm()
        self.act = act()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.act(x)
        return x


class LayerScaler(layers.Layer):
    def __init__(self, init_value: float, dimensions: int):
        super().__init__()
        self.gamma = tf.Variable(
            init_value * tf.ones((dimensions)), trainable=True, dtype=tf.float32
        )

    def call(self, x):
        return self.gamma[None, None, None, ...] * x


class BottleNeckBlock(layers.Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        layer_scaler_init_value: float = 1e-6,
    ):
        super().__init__()
        expanded_features = out_features * expansion
        self.block = tf.keras.Sequential(
            [
                # narrow -> wide (with depth-wise and bigger kernel)
                layers.DepthwiseConv2D(
                    kernel_size=7, padding="same", depth_multiplier=1
                ),
                layers.BatchNormalization(),
                layers.Activation(tf.nn.relu),
                layers.Conv2D(expanded_features, kernel_size=1),
                layers.Activation(tf.nn.relu),
                # wide -> narrow
                layers.Conv2D(out_features, kernel_size=1),
            ]
        )
        self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
       
    def call(self, x: tf.Tensor) -> tf.Tensor:
        res = x
        x = self.block(x)
        x = self.layer_scaler(x)
   
        x += res
        return x


class ConvNexStage(layers.Layer):
    def __init__(self, in_features: int, out_features: int, depth: int, **kwargs):
        super().__init__()
        self.conv = layers.Conv2D(
            out_features,
            kernel_size=2,
            strides=2,
            padding="valid",
            **kwargs,
        )
        self.blocks = [
            BottleNeckBlock(out_features, out_features, **kwargs)
            for _ in range(depth)
        ]

    def call(self, x):
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        return x


class ConvNextStem(layers.Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.conv = layers.Conv2D(
            out_features, kernel_size=4, strides=4, padding="same"
        )
        self.norm = layers.BatchNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x




class ConvNextEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
   
    ):
        super().__init__()
        self.stem = ConvNextStem(in_channels, stem_features)

        in_out_widths = list(zip(widths, widths[1:]))
       

        self.stages = [
            ConvNexStage(stem_features, widths[0], depths[0]),
            *[ConvNexStage(in_features, out_features, depth)
              for (in_features, out_features), depth in zip(
                  in_out_widths, depths[1:]
            )],
        ]

    def call(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x




class ClassificationHead(tf.keras.Sequential):
    def __init__(self):
        super().__init__(layers=[
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation('sigmoid')
        ])


class ConvNext_TF(tf.keras.Model):
    def __init__(
        self,
        in_channels: int,
        stem_features: int,
        depths: List[int],
        widths: List[int],
    ):
        super(ConvNext_TF, self).__init__()

        self.encoder = ConvNextEncoder(in_channels, stem_features, depths, widths)
        self.head = ClassificationHead()

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.head(x)
        return x


if __name__=="__main__":
    image = tf.random.uniform((1, 224, 224, 2))
    classifier = ConvNext_TF(
        in_channels=2,
        stem_features=64,
        depths=[3, 4, 6, 4],
        widths=[256, 512, 1024, 2048],
    )
    print(classifier(image).shape)