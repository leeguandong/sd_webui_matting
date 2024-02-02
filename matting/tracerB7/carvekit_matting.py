import cv2
import torch
import PIL.Image
import numpy as np

from PIL import Image
from typing import Any, Tuple
from PIL.Image import Image as PILImage
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def tracerb7(
        image: np.ndarray,
        result_type: str,
        bg_color: str
):
    # Check doc strings for more information
    seg_net = TracerUniversalB7(device=device, batch_size=1)

    fba = FBAMatting(device=device,
                     input_tensor_size=2048,
                     batch_size=1)

    trimap = TrimapGenerator()

    preprocessing = PreprocessingStub()

    postprocessing = MattingMethod(matting_module=fba,
                                   trimap_generator=trimap,
                                   device=device)

    interface = Interface(pre_pipe=preprocessing,
                          post_pipe=postprocessing,
                          seg_pipe=seg_net)

    image = Image.fromarray(image)
    cutout = interface([image])[0]
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
