# ITMO_EDLM_project

Кратко
------
Проект по квантизации модели CartoonGAN для ускорения модели на инференсе (CPU). Реализованы следующие методы ускорения:
- PTQ (Post Training Quantization)

Репозиторий содержит Docker с инструкцией по запуску

❗Что необходимо❗
---------
- Установить Docker
- Подгрузить веса генератора после клонирования репозитория в директорию weights, сохранив оригинальное название (trained_netG.pth)

Ссылка на оригинальные веса:
[здесь](https://drive.google.com/drive/folders/1d_GsZncTGmMdYht0oUWG9pqvV4UqF_kM?usp=sharing)

Сборка Docker-образа
--------------------
Соберите образ командой:
```bash
docker build -t QuantizedGAN .
```

Запуск контейнера
-----------------
Интерактивный запуск с bash:
```bash
docker run -it --rm QuantizedGAN
```
Для повторения эксперимента переходим в контейнер и запускаем нужный метод по инструкции ниже

После входа в контейнер
-----------------------
- Виртуальное окружение `env` активируется автоматически (но лучше убедиться в этом).
- Рабочая директория: `/app`.
- Запускать Python-скрипты, например:
```bash
python script.py
```

Post-Training Quantization (PTQ)
-------------------------------
Сравнение оригинала с квантованной версией:

<div style="display: flex; gap: 20px;">

  <figure style="text-align: bottom;">
    <img src="gifs/output_o.gif" width="300">
    <figcaption><b>Original version</b></figcaption>
  </figure>

  <figure style="text-align: bottom;">
    <img src="gifs/output_q.gif" width="300">
    <figcaption> <b>Quantized version </b></figcaption>
  </figure>

</div>


Ниже — общая последовательность шагов для запуска PTQ
По итогу выполнения квантизации веса модели сохраняются по указанному пути (save_path)

```
cd src
python quantization/PTQ/post_training_quantization.py --weights_path путь_до_весов --save_path путь_куда_сохранить_веса
--img_size размер_изображения(опционально) --calib_size количество_фейк_изображений(опционально)
```
