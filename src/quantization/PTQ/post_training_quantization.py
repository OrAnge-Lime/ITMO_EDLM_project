from generator import Generator


import torch.quantization as tq
import torch
from torchvision import datasets
import torchvision.transforms as transforms

from tqdm import tqdm

CALIB_SIZE = 32
IMAGE_SIZE = 512
DEVICE = "cpu"

def get_no_aug_transform(size:int = IMAGE_SIZE) -> Generator:
    return transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



def run_post_train_quantization(
        weights_path: str,
        save_path: str = None,
        img_size: int = IMAGE_SIZE,
        calib_size: int = CALIB_SIZE,
):
    """
        Parameters:
        weights_path: Путь до весов модели CartoonGAN (trained_netG.pth)
        save_path: Путь куда сохранить веса модели (например /weights/generator_int8.pth)
        img_size: Размер изображения (по умолчанию 512)
        calib_size: Количество фейковых изображений для калибровки параметров квантизации

        Return:
        Квантизованная модель
    """

    netG = Generator()
    netG.eval()
    netG.load_state_dict(torch.load(weights_path), map_location=torch.device(DEVICE))

    qconfig = tq.get_default_qconfig('qnnpack')
    netG.qconfig = qconfig


    calib_dataset = datasets.FakeData(
        size=calib_size,
        transform=get_no_aug_transform(img_size)
    )
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=4)
    tq.prepare(netG, inplace=True)

    with torch.no_grad():
        for images, _ in tqdm(calib_loader):
            netG(images)

    tq.convert(netG, inplace=True)

    #Check if quantization is successful
    shape = (1, 3, img_size, img_size)
    fake_image = torch.rand(shape)

    try:
        with torch.no_grad():
            netG(fake_image)

    except Exception as e:
        print(e)


    if save_path:
        torch.save(netG.state_dict(), save_path, weights_only=False)

    return netG
