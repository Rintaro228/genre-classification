import json
import os
import math
import librosa

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # в секундах
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path="genres", json_path="data_10.json",
              num_mfcc=10, n_fft=2048, hop_length=512, num_segments=10):
    """Извлекает MFCC из набора музыкальных данных и сохраняет их в файл json в соответствии с ярлыками жанров.
        :param dataset_path: Путь до набора данных
        :param json_path: Путь до будущего файла json с MFCC
        :param num_mfcc: На сколько коэффициентов MFCC разбиваем каждый фрейм.
        :param n_fft: Количество сэмплов, на которых мы используем преобразование Фурье:
        :param hop_length: Смещение между окнами преобразования Фурье
        :param num_segments: На сколько сегментов нужно поделить трэк"""

    # словарь для отсортировывания mapping, labels и MFCC
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # Цикл через все суб-папки в папке genre
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # убедимся, что мы не в папке genres
        if dirpath is not dataset_path:

            # записываем название жанра, то есть название суб-папки
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # обрабатываем все аудилфайлы
            for f in filenames:

                # загружаем аудиофайл
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # обрабатываем все сегменты
                for d in range(num_segments):

                    # считаем начало и конец для каждого сегмента
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # извлечение mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate,
                                                n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # записываем вектор mfcc и номер жанра
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))

    # сохраняем mfcc в json файл
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    dataset_path = input("Введіть шлях до набору даних: ")
    json_path = input("Введіть шлях збереження вихідних даних: ")
    save_mfcc(dataset_path=dataset_path, json_path=json_path)
