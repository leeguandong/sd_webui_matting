import cv2
import numpy as np
import onnxruntime as ort

from PIL import Image
from PIL.Image import Image as PILImage
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union


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


# Get x_scale_factor & y_scale_factor to resize image
def get_scale_factor(im_h, im_w, ref_size):
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / im_w
    y_scale_factor = im_rh / im_h

    return x_scale_factor, y_scale_factor


def putalpha_cutout(img, mask) -> PILImage:
    mask = Image.fromarray(mask)
    img = Image.fromarray(img)
    img.putalpha(mask)
    return img


def modnet(
        image: np.ndarray,
        result_type: str,
        bg_color: str):
    ref_size = 512

    # read image
    # im = image
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = image.copy()

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    im = (im - 127.5) / 127.5

    im_h, im_w, im_c = im.shape
    x, y = get_scale_factor(im_h, im_w, ref_size)

    # resize image
    im = cv2.resize(im, None, fx=x, fy=y, interpolation=cv2.INTER_AREA)

    # prepare input shape
    im = np.transpose(im)
    im = np.swapaxes(im, 1, 2)
    im = np.expand_dims(im, axis=0).astype('float32')

    # Initialize session and get prediction

    # if session is None:
    model_path = str(Path(__file__).resolve().parent)
    # global session
    session = ort.InferenceSession(f"{model_path}/models/modnet.onnx",
                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: im})

    # refine matte
    matte = (np.squeeze(result[0]) * 255).astype('uint8')
    matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation=cv2.INTER_AREA)

    bgcolor = hex_to_rgba(bg_color)
    if bgcolor is not None and result_type == "ReplaceBG":
        cutout = apply_background_color(matte, bgcolor)
    else:
        cutout = putalpha_cutout(image, matte)

    pil_mask = Image.fromarray(matte).convert('L')
    im_rgba = cutout.copy()
    return im_rgba, pil_mask
