import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


def load_data(data_path):
    """ Загружает набор данных из файла json
    :param data_path (str): путь к набору данных
    :return X, y: 3-х мерный массив mfcc и массив ярлыков с номерами жанров"""

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # конвертируем из list в numpy array
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data succesfully loaded!")

    return X, y


def build_cnn_model(input_shape):
    """Генерирует CNN модель
    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # создает топологию нейросети
    model = keras.Sequential()

    # 1 сверточный слой
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2 сверточный слой
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3 сверточный слой
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # преобразуем в flatten(из 2D в 1D) и подаём в Dense слой
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # выходной слой
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def build_mlp_model(input_shape):
    """Генерирует MLP модель
        :param input_shape (tuple): Shape of input set
        :return model: MLP model
        """
    # Строим архитектуру
    model = keras.Sequential([

        # Делаем входной слой
        keras.layers.Flatten(input_shape=input_shape),

        # Первый скрытый слой dense - свзязанный со всеми нейронами следующего слоя
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2 скрытый слой
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3 скрытый слой
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # Выходной слой
        keras.layers.Dense(10, activation='softmax')
    ])

    return model


def plot_history(history):

    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":

    data_path = input("Введіть шлях до вхідних даних (json файл): ")

    # загружаем данные
    X, y = load_data(data_path)

    # делим данные на тренировочные и тестовые
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model_type = input("Введіть вид нейромережі. Введіть mlp для застосування багатошарового перцептрону або cnn для"
                       " застосування згорткової нейронної мережі: ")

    if model_type == "cnn":

        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        model = build_cnn_model(input_shape)

    elif model_type == "mlp":

        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_mlp_model(input_shape)

    optimiser = input("Введіть оптимізатор (adam or sgd): ")

    if optimiser == "adam":

        optimiser = keras.optimizers.Adam(learning_rate=0.0001)

    elif optimiser == "sgd":

        optimiser = keras.optimizers.SGD(learning_rate=0.001)

    # компилируем модель
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    epochs = int(input("Введіть кількість епох: "))

    # тренируем модель
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)

    # характеризуем модель
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # выводим графики
    plot_history(history)