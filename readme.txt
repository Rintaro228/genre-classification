Файли на цьому диску:
docs\MAN_Oblast_Kharchenko.pdf -- наукова робота у форматі PDF;
docs\MAN_Oblast_Kharchenko.docx -- наукова робота у форматі Word;
src\preparing_data.py -- програма, яка перетворює дані у MFCC;
src\main.py -- основна програма, яка створює та тренерує нейромережу нейромережу.

--------------------------------------------------------------------------------------------------------
1. Скачайте набір даних (папка "genres") за посиланням у Google Drive -- https://drive.google.com/drive/folders/1fdzyHKPgSGZlPUjKtnvbbblZ5yb9vWRf?usp=sharing 
2. Запустіть "preparing_data.py". Ця програма перетворює початковий набір даних у json файл, 
   який потім буде потрібен для нейромережі. Після цього запам'ятайте шлях до json файлу, який був створений.
3. Запустіть "main.py". Це і є основна програма, у якій знаходиться нейромережа.
   На вихід вона подає графік точності та помилки за тренувальними та тестовими даними за епохами. 
   Також вона подає точність на тестових даних після останньої епохи.
--------------------------------------------------------------------------------------------------------
Бібліотеки, які були використовувані:
import json
import os
import math
import librosa
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
Я використовувал Python 3.8, на інших версіях Python негарантована працездатність програм.