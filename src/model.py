from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, ReLU, Dropout, merge
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def build_model(input):
    # L1
    x = Conv2D(64, (15, 15), strides=(3, 3))(input)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2)) (x)
    # L2
    x = Conv2D(128, (5, 5), strides=(1, 1))(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    # L3
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    # L4
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    # L5
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Flatten()(x)
    # L6
    x = Dense(512)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    # L7
    x = Dense(512)(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(340)(x)
    return x


class SketchNet:
    def __init__(self):
        self.model = None

    def construct_model(self):
        input1 = Input(shape=(225, 225, 1))
        input2 = Input(shape=(225, 225, 1))
        input3 = Input(shape=(225, 225, 1))
        input4 = Input(shape=(225, 225, 1))
        input5 = Input(shape=(225, 225, 1))


        y1 = build_model(input1)
        y2 = build_model(input2)
        y3 = build_model(input3)
        y4 = build_model(input4)
        y5 = build_model(input5)

        out = merge.average([y1, y2, y3, y4, y5])
        # out = keras.layers.concatenate([y1, y2], axis=0)

        self.model = Model(inputs=[input1, input2, input3, input4, input5], outputs=out)
        print(self.model.summary())

    def compile(self):


        self.model.compile(optimizer=Adam(lr=0.002),
                        loss='categorical_crossentropy',
                        metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])




if __name__ == "__main__":
    sketchnet = SketchNet()
    sketchnet.construct_model()
    sketchnet.compile()
