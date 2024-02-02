import sys
from pathlib import Path

# import pdb;pdb.set_trace()
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from PIL import Image
from typing import Any, Tuple
from PIL.Image import Image as PILImage
from transparent_background import Remover


def hex_to_rgba(hex_code):
    # 去除 # 号
    hex_code = hex_code.lstrip('#')
    # 将十六进制颜色代码转换为 RGB 三个数值
    rgb = tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))
    # 将 RGB 转换为 RGBA
    rgba = rgb + (255.0,)
    return rgba


def apply_background_color(img, color: Tuple[int, int, int, int]) -> PILImage:
    r, g, b, a = color
    colored_image = Image.new("RGBA", img.size, (int(r), int(g), int(b), int(a)))
    colored_image.paste(img, mask=img)

    return colored_image


def inspyrenet(
        image: np.ndarray,
        model_id: str,
        threshold: float,
        result_type: str,
        bg_color: str
):
    # Load model
    remover = Remover(mode=model_id)  # default setting
    # remover = Remover(mode='fast', jit=True, device='cuda:0', ckpt='~/latest.pth')  # custom setting
    # remover = Remover(mode='base-nightly')  # nightly release checkpoint

    # Usage for image
    img = Image.fromarray(image).convert("RGB")

    # out = remover.process(img)  # default setting - transparent background
    # out = remover.process(img, type='rgba')  # same as above
    # out = remover.process(img, type='map')  # object map only
    # out = remover.process(img, type='green')  # image matting - green screen
    # out = remover.process(img, type='white')  # change backround with white color
    # out = remover.process(img, type=[255, 0, 0])  # change background with color code [255, 0, 0]
    # out = remover.process(img, type='blur')  # blur background
    # out = remover.process(img, type='overlay')  # overlay object map onto the image
    # out = remover.process(img, type='samples/background.jpg')  # use another image as a background

    cutout = remover.process(img, threshold=threshold)  # use threhold parameter for hard prediction.

    cutout = np.array(cutout)

    alpha_channel = cutout.copy()[:, :, 3]
    alpha_channel[alpha_channel == 0] = 0
    alpha_channel[alpha_channel != 0] = 255

    # 色彩空间转换和阈值化操作得到黑白mask图
    gray = cv2.cvtColor(cutout, cv2.COLOR_RGB2GRAY)
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # 将alpha通道作为mask图的第四个通道，方便后续处理
    masked_image = cv2.merge((mask, mask, mask, alpha_channel))
    masked_image = Image.fromarray(masked_image).convert('L')

    bgcolor = hex_to_rgba(bg_color)
    if bgcolor is not None and result_type == "ReplaceBG":
        cutout = apply_background_color(cutout, bgcolor)
    cutout = Image.fromarray(cutout)

    return cutout, masked_image
