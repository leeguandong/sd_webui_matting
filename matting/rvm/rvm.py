import av
import os
import pims
import numpy as np
import torch
import onnxruntime as ort

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple, Union, Type
from torch.hub import download_url_to_file


def download_models_rvm(model_id):
    if "rvm_resnet50_human" in model_id:
        url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx"
    else:
        url = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
    models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    checkpoint = os.path.join(models_dir, model_id)

    if not os.path.isfile(checkpoint):
        try:
            download_url_to_file(url, checkpoint)
        except Exception as e:
            print(f"{str(e)}")
    else:
        print("Model already exists")
    return checkpoint


class ImageSequenceReader(Dataset):
    def __init__(self, path, transform=None):
        self.files = [path]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = self.files[idx]
        if self.transform is not None:
            return self.transform(img)
        return img


def convert_video(session: ort.InferenceSession,
                  input_source: np.ndarray,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = 0.25,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = 'cpu',
                  dtype: Type[Union[np.single, np.half]] = np.float32):
    # assert downsample_ratio is None or (
    #         downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    # assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    # assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    # assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    # assert num_workers >= 0, 'Number of workers must be >= 0'

    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    img = Image.fromarray(input_source)
    source = ImageSequenceReader(img, transform)
    if downsample_ratio is None:
        downsample_ratio = auto_downsample_ratio(*source[0].shape[1:])
    reader = DataLoader(source, batch_size=1, pin_memory=True, num_workers=1)

    # Inference
    if device.lower() == "cuda":
        io = session.io_binding()
        # Create tensors on CUDA.
        rec = [ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=dtype), 'cuda')] * 4
        downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([downsample_ratio], dtype=np.float32), 'cuda')
        # Set output binding.
        for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
            io.bind_output(name, 'cuda')
    else:
        io = None
        rec = [np.zeros([1, 1, 1, 1], dtype=dtype)] * 4  # Must match dtype of the model.
        downsample_ratio = np.array([downsample_ratio], dtype=np.float32)  # dtype always FP32

    img_rgba, mask = None, None
    for src in reader:
        src = src.cpu().numpy()  # torch.Tensor -> np.ndarray CPU [B,C,H,W] for ONNX file
        if device.lower() == "cuda" and io is not None:
            io.bind_cpu_input('src', src)
            io.bind_ortvalue_input('r1i', rec[0])
            io.bind_ortvalue_input('r2i', rec[1])
            io.bind_ortvalue_input('r3i', rec[2])
            io.bind_ortvalue_input('r4i', rec[3])
            io.bind_ortvalue_input('downsample_ratio', downsample_ratio)

            session.run_with_iobinding(io)
            fgr, pha, *rec = io.get_outputs()

            # Only transfer `fgr` and `pha` to CPU.
            fgr = fgr.numpy()
            pha = pha.numpy()
        else:
            fgr, pha, *rec = session.run([], {
                'src': src,
                'r1i': rec[0],
                'r2i': rec[1],
                'r3i': rec[2],
                'r4i': rec[3],
                'downsample_ratio': downsample_ratio
            })

        fgr = torch.from_numpy(fgr).unsqueeze(1)  # [B,T=1,C=3,H,W]
        pha = torch.from_numpy(pha).unsqueeze(1)  # [B,T=1,C=1,H,W]
        fgr = fgr * pha.gt(0)
        com = torch.cat([fgr, pha], dim=-3)

        img_rgba = to_pil_image(com[0][0])
        mask = to_pil_image(pha[0][0])

    return img_rgba, mask


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


def hex_to_rgba(hex_code):
    # 去除 # 号
    hex_code = hex_code.lstrip('#')
    # 将十六进制颜色代码转换为 RGB 三个数值
    rgb = tuple(int(hex_code[i:i + 2], 16) for i in (0, 2, 4))
    # 将 RGB 转换为 RGBA
    rgba = rgb + (255.0,)
    return rgba


def apply_background_color(img, color: Tuple[int, int, int, int]):
    r, g, b, a = color
    colored_image = Image.new("RGBA", img.size, (int(r), int(g), int(b), int(a)))
    colored_image.paste(img, mask=img)

    return colored_image


class Converter:
    def __init__(self, checkpoint: str, device: str, dtype: str, result_type: str, bg_color: str):
        assert dtype in ("fp16", "fp32")
        # 此处可以用原生的torch推理，就没这么多问题了
        self.session = ort.InferenceSession(checkpoint)
        self.device = device
        self.dtype = np.float32 if dtype == "fp32" else np.float16
        self.result_type = result_type
        self.bg_color = bg_color

    def convert(self, *args, **kwargs):
        img_rgba, mask = convert_video(self.session, device=self.device, dtype=self.dtype, *args, **kwargs)
        bgcolor = hex_to_rgba(self.bg_color)
        if bgcolor is not None and self.result_type == "ReplaceBG":
            mask = apply_background_color(mask, bgcolor)
        return img_rgba, mask

# https://huggingface.co/spaces/peterkros/videomatting/blob/main/app.py
