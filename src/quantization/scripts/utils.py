import mimetypes
from PIL import Image
import subprocess
import tempfile
from tqdm import tqdm
import os 
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def get_no_aug_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]


def predict_file(model, input_path, output_path, batch_size):
    # File is image
    if mimetypes.guess_type(input_path)[0].startswith("image"):
        image = Image.open(input_path).convert('RGB')
        predicted_image = predict_images([image], model)[0]
        predicted_image.save(output_path)
    
    # File is video
    elif mimetypes.guess_type(input_path)[0].startswith("video"):
    # Создаём временную директорию (она удалится автоматически при выходе из with)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Путь-шаблон для ffmpeg (кадры положатся в tmpdir)
            out_template = os.path.join(tmpdir, "frame_%07d.png")
    
            # Команда ffmpeg — передаём аргументы как список (без shell),
            # чтобы %07d корректно обрабатывался ffmpeg'ом
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", input_path,
                "-vsync", "0",            # надёжное сохранение всех кадров
                "-start_number", "1",     # нумерация с 1 (можно поставить 0)
                "-loglevel", "error",     # можно убрать, если хочешь видеть предупреждения
                out_template
            ]
    
            try:
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                # Выведем stderr ffmpeg для диагностики
                stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
                raise RuntimeError(f"ffmpeg failed: {stderr}") from e
    
            # Получаем список сохранённых кадров и сортируем по номеру в имени
            frame_paths = glob.glob(os.path.join(tmpdir, "frame_*.png"))
    
            # Если не найдено кадров — выбросим понятную ошибку
            if not frame_paths:
                raise FileNotFoundError(f"No frames were extracted to {tmpdir}. Check ffmpeg output and input_path.")
    
            # Сортировка по числовому индексу в имени файла
            def _frame_key(path):
                m = re.search(r"(\d+)\.png$", path)
                return int(m.group(1)) if m else -1
    
            frame_paths.sort(key=_frame_key)
    
            # Разбиваем на батчи и обрабатываем
            batches = [*divide_chunks(frame_paths, batch_size)]
            for path_chunk in tqdm(batches):
                imgs = [Image.open(p) for p in path_chunk]
                imgs = predict_images(imgs, model)  # предполагаем, что возвращает список PIL.Image
                for path, img in zip(path_chunk, imgs):
                    img.save(path)

            fps = 30.0

            output_video = output_path
    
            ffmpeg_encode = [
                "ffmpeg",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "frame_%07d.png"),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-loglevel", "error",
                output_video
            ]

            subprocess.run(ffmpeg_encode, check=True)
    
            print("Готово! Видео сохранено в:", output_video)
    else:
        raise IOError("Invalid file extension.")


def predict_images(image_list, model):
    trf = get_no_aug_transform()
    image_list = torch.from_numpy(np.array([trf(img).numpy() for img in image_list]))

    with torch.no_grad():
        generated_images = model(image_list)
    generated_images = inv_normalize(generated_images)

    pil_images = []
    for i in range(generated_images.size()[0]):
        generated_image = generated_images[i].cpu()
        pil_images.append(TF.to_pil_image(generated_image))
    return pil_images


def inv_normalize(img):
    # Adding 0.1 to all normalization values since the model is trained (erroneously) without correct de-normalization
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    img = img.clamp(0, 1)
    return img