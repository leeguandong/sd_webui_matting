import sys
from pathlib import Path

# import pdb;pdb.set_trace()
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from hashlib import sha1
from PIL import Image
from paddleseg.cvlibs import manager, Config

# from paddleseg.utils import load_entire_model
manager.BACKBONES._components_dict.clear()
manager.TRANSFORMS._components_dict.clear()

import ppmatting as ppmatting
from ppmatting.core import predict, ppmatting_predict
from ppmatting.utils import load_pretrained_model, estimate_foreground_ml

model_names = [
    "ppmatting-hrnet_w18-human_512",
    "ppmatting-hrnet_w18-human_1024",
    "ppmatting-hrnet_w18-human_2048",
    "ppmatting-hrnet_w48-composition",
    "ppmatting-hrnet_w48-distinctions",
    "ppmattingv2-stdc1-human_512",
    "modnet-mobilenetv2",
    "modnet-hrnet_w18",
    "modnet-resnet50_vd",
]

model_dict = {
    name: None
    for name in model_names
}

last_result = {
    "cache_key": None,
    "algorithm": None,
}


def image_ppmatting(
        image: np.ndarray,
        result_type: str,
        bg_color: str,
        algorithm: str,
        morph_op: str,
        morph_op_factor: float,
):
    image = np.ascontiguousarray(image)
    cache_key = sha1(image).hexdigest()
    if cache_key == last_result["cache_key"] and algorithm == last_result["algorithm"]:
        alpha = last_result["alpha"]
    else:
        path = str(Path(__file__).resolve().parent)
        cfg = Config(f"{path}/configs/{algorithm}.yml")
        if model_dict[algorithm] is not None:
            model = model_dict[algorithm]
        else:
            model = cfg.model
            load_pretrained_model(model, f"{path}/models/{algorithm}.pdparams")
            model.eval()
            model_dict[algorithm] = model

        transforms = ppmatting.transforms.Compose(cfg.val_transforms)

        alpha = ppmatting_predict(
            model,
            transforms=transforms,
            image=image,
        )
        last_result["cache_key"] = cache_key
        last_result["algorithm"] = algorithm
        last_result["alpha"] = alpha

    alpha = (alpha * 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    if morph_op == "dilate":
        alpha = cv2.dilate(alpha, kernel, iterations=int(morph_op_factor))
    elif morph_op == "erode":
        alpha = cv2.erode(alpha, kernel, iterations=int(morph_op_factor))

    alpha = (alpha / 255).astype(np.float32)
    image = (image / 255.0).astype("float32")
    fg = estimate_foreground_ml(image, alpha)

    if result_type == "ReplaceBG" and bg_color is not None:
        bg_r = int(bg_color[1:3], base=16)
        bg_g = int(bg_color[3:5], base=16)
        bg_b = int(bg_color[5:7], base=16)

        bg = np.zeros_like(fg)
        bg[:, :, 0] = bg_r / 255.
        bg[:, :, 1] = bg_g / 255.
        bg[:, :, 2] = bg_b / 255.

        result = alpha[:, :, None] * fg + (1 - alpha[:, :, None]) * bg
        result = np.clip(result, 0, 1)
    else:
        result = np.concatenate((fg, alpha[:, :, None]), axis=-1)

    mask = Image.fromarray((alpha * 255).astype(np.uint8))
    result = Image.fromarray((result*255).astype(np.uint8))
    return result, mask
