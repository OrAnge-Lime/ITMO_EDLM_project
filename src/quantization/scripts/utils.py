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
        # Create temp folder for storing frames as images
        temp_dir = tempfile.TemporaryDirectory()
        # Extract frames from video
        subprocess.run(f"ffmpeg -i \"{input_path}\" -loglevel error -stats \"{os.path.join(temp_dir.name, 'frame_%07d.png')}\"")
        # Process images with model
        frame_paths = listdir_fullpath(temp_dir.name)
        batches = [*divide_chunks(frame_paths, batch_size)]
        for path_chunk in tqdm(batches):
            imgs = [Image.open(p) for p in path_chunk]
            imgs = predict_images(imgs)
            for path, img in zip(path_chunk, imgs):
                img.save(path)
        # Get video frame rate
        frame_rate = subprocess.check_output(f"ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate \"{input_path}\"")
        frame_rate = eval(frame_rate.split()[0]) # Dirty eval
        # Combine frames with original audio
        subprocess.run(f"ffmpeg -y -r {frame_rate} -i \"{os.path.join(temp_dir.name, 'frame_%07d.png')}\" -i \"{input_path}\" -map 0:v -map 1:a? -loglevel error -stats \"{output_path}\"")
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