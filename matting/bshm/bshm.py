import cv2
import numpy as np
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from typing import Any, Tuple
from PIL.Image import Image as PILImage


def hex_to_rgba(hex_code):
    # 去除 # 号
    hex_code = hex_code.lstrip('#')
    # 将十六进制颜色代码转换为 RGB 三个数值
    rgb = tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))
    # 将 RGB 转换为 RGBA
    rgba = rgb + (255.0,)
    return rgba


def apply_background_color(img: np.ndarray, color: Tuple[int, int, int, int]) -> PILImage:
    img = Image.fromarray(img)
    r, g, b, a = color
    colored_image = Image.new("RGBA", img.size, (int(r), int(g), int(b), int(a)))
    colored_image.paste(img, mask=img)

    return colored_image


# new_model_id = None
# universal_matting = pipeline(Tasks.universal_matting, model="damo/cv_unet_universal-matting")

def putalpha_cutout(img, mask) -> PILImage:
    mask = Image.fromarray(mask)
    img = Image.fromarray(img)
    img.putalpha(mask)
    return img


def bshm(
        image: np.ndarray,
        model_id: str,
        result_type: str,
        bg_color: str
):
    # global new_model_id
    # if model_id != new_model_id:
    #     universal_matting = pipeline(Tasks.universal_matting, model=model_id)
    #     new_model_id = model_id

    universal_matting = pipeline(Tasks.universal_matting, model=model_id)
    result = universal_matting(image)
    cutout = result[OutputKeys.OUTPUT_IMG]

    alpha_channel = cutout.copy()[:, :, 3]
    alpha_channel[alpha_channel == 0] = 0
    alpha_channel[alpha_channel != 0] = 255

    # 色彩空间转换和阈值化操作得到黑白mask图
    gray = cv2.cvtColor(cutout, cv2.COLOR_RGB2GRAY)
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # 将alpha通道作为mask图的第四个通道，方便后续处理
    masked_image = cv2.merge((mask, mask, mask, alpha_channel))
    masked_image = Image.fromarray(masked_image).convert('L')

    #     image_ori = Image.fromarray(image)

    #     # 获取 Alpha 通道数据
    #     mask = (cutout > 0).astype(np.uint8) * 255  # 将 Alpha 通道数据转换为二进制格式

    #     # 将掩膜转换为 PIL 图像对象
    #     mask_image = Image.fromarray(mask)
    #     import pdb;pdb.set_trace()
    #     # 可选：将掩膜与原图叠加，以检查是否正确生成掩膜
    #     mask_image_rgba = Image.merge("RGBA", [mask_image] * 3)
    #     masked_image = Image.composite(image_ori, mask_image_rgba, mask_image)
    bgcolor = hex_to_rgba(bg_color)
    if bgcolor is not None and result_type == "ReplaceBG":
        cutout = apply_background_color(cutout, bgcolor)
    cutout = Image.fromarray(cutout)

    return cutout, masked_image
