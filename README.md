# train_number_detection


Решение команды **ГВОЗДОМЕТ**

### Структура проекта

1. model.py - код нашей модели машинного обучения для детекции и распознования цифр.
2. Api.ipynb - API-интерфейс Flask, который получает фотографию с помощью API, и возвращает JSON-объект с распознанами цифрами.
3. train_num_yolov5_weights.pt - веса модели для детекции цифр.

### Running the project
Перед запуском надо установить все необходимые библиотеки
```pip install -r requirements.txt```

<br />

**При возникновении ошибок из-за отсутствия установленных библиотек, выполните эти команды в консоли**
<br />

```
pip install glob
pip install torch
pip install utils
pip install opencv-python
pip install pillow
pip install pandas
pip install numpy
pip install easyocr
pip install Flask
```

# Инструкция по работе с API

1. Поместите фотографии для распознования в папку *images*

2. Запустите файл **Api.ipynb**

3. Перейдите по ссылке:         

```127.0.0.1:5000/pred/'имя файла с изображением без расширения'```<br /><br />

Пример для изображения **42343657.jpg**
```
http://127.0.0.1:5000/pred/42343657
```


# Пример работы

<h3>На вход модель получает фотографию:<h3>

![42343657](https://github.com/vlyrdv/train_number_detection/assets/61351039/db93b45f-f4d8-4f04-8289-0ec6bee2a757)

<br />

<h3>Дальше модель вырезает номер: </h3>

<br />

![itog2](https://github.com/vlyrdv/train_number_detection/assets/61351039/a5ef67a5-7129-4589-832a-474fc01edc2d)



<h3>И возвращает JSON объект:</h3>

```python
{"images/42343657.jpg": {
                        "is_correct": 1,
                        "number": 42343657,
                        "time": "14.10.2023 20:45",
                        "type": 1}}
```

**is_correct** - оценка корректности распознанного номера (1 - верно, 0 - не верно)

**number** - номер вагона

**time** - время в которое происходило распознание

**type** - существование номера на фото (1 - номер найден, 0 - не найден)

# Пример работы с Web интерфейсом по ссылке:
```http://127.0.0.1:5000/pred/42343657```

![skreen cast and gif](https://github.com/vlyrdv/train-number-detection/assets/61351039/ecf94a22-2450-4e66-ab5d-defff08ce1bb)



# Принцип работы модели

1. Изменяет размер подаваемого изображения на (width: 640, hight: 640)

2. Детектирует номер вагона на вышеупомянутом фото с помощью предобученных весов YOLOV5

3. Вырезает найденный объект изображения и дальше продолжает работать уже с ним

4. Превращает его в серо-черно-белое

5. Детектирует цифры с помощью **easyocr**





В проекте участвовали 
- <h4><a href="https://github.com/vlyrdv">Урядов Валерий</a></h4>
- <h4><a href="https://github.com/abrikosmna">Мокийчук Никита</a></h4>
- <h4><a href="https://github.com/GermanKek-lab">Кек Герман</a></h4>

