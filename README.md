# ITMO_EDLM_project

Оригинал: https://github.com/FilipAndersson245/cartoon-gan

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
    <img src="gifs/output_o.gif" width="256">
    <figcaption><b>Original version</b></figcaption>
  </figure>

  <figure style="text-align: bottom;">
    <img src="gifs/output_q.gif" width="256">
    <figcaption> <b>Quantized version </b></figcaption>
  </figure>

</div>

***
- В директории PTQ есть ноутбук с примером квантизации
- Ниже — общая последовательность шагов для запуска PTQ из функции run_post_training_quantization
- По итогу выполнения квантизации веса модели сохраняются по указанному пути (save_path)

```
cd src
python quantization/PTQ/post_training_quantization.py --weights_path путь_до_весов --save_path путь_куда_сохранить_веса
--img_size размер_изображения(опционально) --calib_size количество_фейк_изображений(опционально)
```


Structured pruning
-------------------------------
Для реализации прунинга была использована утилита torch_pruning с дальнейшим дообучением модели на [Real to Ghibli Image dataset](https://www.kaggle.com/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images). Ниже представлены примеры генерации оригинальной и вариантов сжатой моделей:

Оригинальная модель CartoonGAN (11M params):
<img width="676" height="359" alt="image_2025-11-22_13-50-35" src="https://github.com/user-attachments/assets/55f13ee7-aef5-4436-8138-0f5533bfa0fd" />

Reduction: 11.7% 9,8M params:

<img width="676" height="359" alt="image_2025-11-22_13-49-19" src="https://github.com/user-attachments/assets/34d5ff9e-cc4e-4e72-9abe-30ea3005b654" />

Reduction: 22.6% 8,6M params:

<img width="676" height="359" alt="image_2025-11-22_13-49-47" src="https://github.com/user-attachments/assets/b5f61852-f8bb-4cec-ab4e-0cd5559ab846" />

Полный пайплайн прунинга с дальнейшим дообучением представлен в ветке `prune` в ноутбуке `prune.ipynb`
