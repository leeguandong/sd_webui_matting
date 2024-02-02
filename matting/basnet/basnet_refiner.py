import sys
from pathlib import Path

# import pdb;pdb.set_trace()
sys.path.append(str(Path(__file__).resolve().parent))

import cv2
import torch
import numpy as np
import onnxruntime as ort

from skimage import measure
from PIL import Image
from PIL.Image import Image as PILImage
from pathlib import Path
from collections import OrderedDict
from typing import Any, List, Optional, Tuple, Union

from app import BASNet, m_net2


class Matting():
    def __init__(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        self.GPU = torch.cuda.is_available()
        torch.set_grad_enabled(False)
        model_path = str(Path(__file__).resolve().parent)

        m_model_path = f'{model_path}/models/mmodel_64.pth'
        b_model_path = f'{model_path}/models/bmodel_3416.pth'
        if self.GPU:
            bckpt = torch.load(b_model_path)
            mckpt = torch.load(m_model_path)
        else:
            bckpt = torch.load(b_model_path, map_location=lambda storage, loc: storage)
            mckpt = torch.load(m_model_path, map_location=lambda storage, loc: storage)
        self.bmodel = BASNet.BASNet(3, 1)
        self.bmodel.load_state_dict(bckpt)
        self.mmodel = m_net2.Net()
        new_state_dict = OrderedDict()
        for k, v in mckpt['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        self.mmodel.load_state_dict(new_state_dict)
        if self.GPU:
            self.bmodel.cuda()
            self.mmodel.cuda()
        self.mmodel.eval()
        self.trimap_scale = 0.02
        self.size = (480, 480)

    def refine_alpha(self, alpha, keep_value=5, keep_rate=0.15):
        img_01 = np.uint8(np.where(alpha < keep_value, 0, 1))
        label_image = measure.label(img_01, connectivity=2)
        labels = measure.regionprops(label_image)
        areas = [label.area for label in labels]
        if len(areas) != 0:
            max_area = sorted(areas)[-1]
            output = np.zeros_like(img_01)
            for label in labels:
                if label.area > max_area * keep_rate:
                    index = label.label
                    index_image = np.where(label_image == index, 1, 0)
                    output = output + index_image
            alpha = output * alpha
            return alpha
        else:
            return alpha

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d - mi) / (ma - mi)
        return dn

    def gen_trimap(self, msk, scale=0.01):
        h, w = msk.shape
        if scale >= 1:
            kernel_size = scale
        else:
            kernel_size = int(scale * (h + w) / 2)
        dmin = 0  # hyper parameter
        emax = 255 - dmin  # hyper parameter
        erode_kernel_size = kernel_size
        dilate_kernel_size = kernel_size
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_kernel_size, erode_kernel_size))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel_size, dilate_kernel_size))
        eroded_alpha = cv2.erode(msk, erode_kernel)
        dilated_alpha = cv2.dilate(msk, dilate_kernel)
        res = dilated_alpha.copy()
        res[((dilated_alpha == 255) & (eroded_alpha == 0))] = 128
        return res

    def predict(self, image, background_color, exist_alpha=None, result_type="RemoveBG"):
        h, w, c = image.shape
        image_resized = cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC)
        used_img = image_resized / 255.0
        used_img = torch.from_numpy(used_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
        if self.GPU:
            used_img = used_img.cuda()
        else:
            used_img = used_img
        d1, _, _, _, _, _, _, _ = self.bmodel(used_img)
        pred = d1[0, 0, :, :]
        pred = self.normPRED(pred)
        if self.GPU:
            pred = pred.cpu().data.numpy()
        else:
            pred = pred.data.numpy()
        mask_pred = 255 * np.where(pred > 0.5, 1, 0)
        trimap = self.gen_trimap(np.float32(mask_pred), scale=self.trimap_scale)
        used_trimap = torch.from_numpy(trimap[np.newaxis, np.newaxis, :, :] / 255.)

        if self.GPU:
            used_trimap = used_trimap.cuda()
        else:
            used_trimap = used_trimap

        alpha_src = self.mmodel(used_img, used_trimap)
        _, _, _, alpha4_src = alpha_src
        alpha = torch.sigmoid(alpha4_src[0])[0] * 255.
        if self.GPU:
            alpha_np = alpha.cpu().data.numpy()
        else:
            alpha_np = alpha.data.numpy()
        alpha = self.refine_alpha(alpha_np)
        trimap_255 = np.where(trimap == 255, 1, 0)
        alpha = (alpha / 255. + trimap_255)
        alpha = 255. * np.where(alpha > 1, 1, alpha)
        alpha = np.uint8(np.clip(np.ceil(cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)), 0, 255))
        alpha = np.where(alpha > 245, 255, alpha)
        alpha = np.where(alpha < 10, 0, alpha)

        if exist_alpha is not None:
            alpha = np.where(exist_alpha == 0, 0, alpha).astype(np.uint8)

        if background_color is not None and result_type == "ReplaceBG":
            color = np.zeros((h, w, 3))
            color[..., 0] = background_color[2]
            color[..., 1] = background_color[1]
            color[..., 2] = background_color[0]
            alpha_3c = np.stack([alpha, alpha, alpha], axis=-1) / 255.
            foreground = alpha_3c * image
            bg = (1 - alpha_3c) * color
            output = foreground + bg
        else:
            # output = np.concatenate((foreground, alpha[:, :, np.newaxis]), axis=2)
            output = np.concatenate((image, alpha[:, :, np.newaxis]), axis=2)

        return output, alpha


matting = Matting()


def hex_to_rgba(hex_code):
    # 去除 # 号
    hex_code = hex_code.lstrip('#')
    # 将十六进制颜色代码转换为 RGB 三个数值
    rgb = tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))
    return rgb


def basnet_refiner(
        image: np.ndarray,
        result_type: str,
        bg_color: str
):
    exist_alpha = None

    shape_info = image.shape
    # img = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    img = image

    if len(shape_info) == 3 and shape_info[2] == 4:
        h, w = shape_info[:2]
        exist_alpha = img[:, :, 3]
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        # img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img = image.copy()
        img = img * (exist_alpha[:, :, np.newaxis] / 255.) + white_bg * ((1 - exist_alpha / 255.)[:, :, np.newaxis])
        img = img.astype(np.uint8)

    else:
        # img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img = image.copy()

    bgcolor = hex_to_rgba(bg_color)
    cutout, alpha = matting.predict(img, bgcolor, exist_alpha=exist_alpha, result_type=result_type)

    pil_mask = Image.fromarray(alpha).convert('L')
    cutout = Image.fromarray(cutout)
    im_rgba = cutout.copy()
    return im_rgba, pil_mask
