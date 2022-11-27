import numpy as np
import pandas as pd


from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


df = pd.read_csv("/content/drive/My Drive/fer2013.csv")
print(df.shape)
df.head()

emotion_label_to_text = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happiness",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}

INTERESTED_LABELS = [3, 4]
df = df[df.emotion.isin(INTERESTED_LABELS)]

img_array = df.pixels.apply(
    lambda x: np.array(x.split(" ")).reshape(48, 48, 1).astype("float32")
)
img_array = np.stack(img_array, axis=0)

le = LabelEncoder()
img_labels = le.fit_transform(df["emotion"])
img_labels = np_utils.to_categorical(img_labels)

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

X_train, X_test, y_train, y_test = train_test_split(
    img_array,
    img_labels,
    shuffle=True,
    stratify=img_labels,
    test_size=0.2,
    random_state=42,
)

img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]

X_train = X_train / 255.0
X_test = X_test / 255.0


def build_net(optim):
    net = Sequential(name="DCNN")

    net.add(
        Conv2D(
            filters=256,
            kernel_size=(5, 5),
            input_shape=(img_width, img_height, img_depth),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_1",
        )
    )
    net.add(BatchNormalization(name="batchnorm_1"))
    net.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_2",
        )
    )
    net.add(BatchNormalization(name="batchnorm_2"))
    net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1"))
    net.add(Dropout(0.4, name="dropout_1"))
    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_3",
        )
    )
    net.add(BatchNormalization(name="batchnorm_3"))
    net.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_4",
        )
    )
    net.add(BatchNormalization(name="batchnorm_4"))
    net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_2"))
    net.add(Dropout(0.4, name="dropout_2"))
    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_5",
        )
    )
    net.add(BatchNormalization(name="batchnorm_5"))
    net.add(
        Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_6",
        )
    )
    net.add(BatchNormalization(name="batchnorm_6"))
    net.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_3"))
    net.add(Dropout(0.5, name="dropout_3"))
    net.add(Flatten(name="flatten"))
    net.add(
        Dense(128, activation="elu", kernel_initializer="he_normal", name="dense_1")
    )
    net.add(BatchNormalization(name="batchnorm_7"))
    net.add(Dropout(0.6, name="dropout_4"))
    net.add(Dense(num_classes, activation="softmax", name="out_layer"))

    net.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    net.summary()

    return net


early_stopping = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.00005,
    patience=10,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=7, min_lr=1e-7, verbose=1
)

callbacks = [early_stopping, lr_scheduler]

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
)
train_datagen.fit(X_train)

batch_size = 64
epochs = 75
optims = [
    optimizers.Nadam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam"
    ),
    optimizers.Adam(0.001),
]

model = build_net(optims[1])
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
)

model.save("model.h5")