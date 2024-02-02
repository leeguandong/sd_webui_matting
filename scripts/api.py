# api
import sys

sys.path.append("/root/autodl-tmp/stable-diffusion-webui/extensions/sd_webui_matting")

import os
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
from PIL import Image

from matting.rembg.session_factory import new_session
from matting.rembg.bg import remove_matting
from matting.pp.pp import image_ppmatting
from matting.modnet.modnet import modnet
from matting.rvm.rvm import Converter, download_models_rvm
from matting.bshm.bshm import bshm
from matting.tracerB7.carvekit_matting import tracerb7
from matting.inspyrenet.transparent_bg import inspyrenet
from matting.basnet.basnet_refiner import basnet_refiner
from matting.basnet.basnet_onestage import basnet
from matting.file_manager import file_manager as file_manager_matting

rembg_model_ids = [
    'u2net',  # 用于一般用例的预训练模型
    'u2netp',  # u2net模型的轻量级版本
    'u2net_human_seg',  # 用于人工分割的预训练模型
    'u2net_cloth_seg',  # 从人类肖像中用于布料解析的预训练模型。这里的衣服被解析为3类：上半身，下半身和全身。在rembg中支持的并不好，目前只能通过使用蒙版的形式显示
    'silueta',  # 与u2net相同，但大小减少到43Mb
    'isnet-general-use',  # 用于一般用例的新预训练模型
    'isnet-anime',  # 动漫角色的高精度分割
]

ppmatting_model_ids = [
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

bshm_model_ids = [
    "damo/cv_unet_universal-matting",
    "damo/cv_unet_image-matting"
]

rvm_model_ids = [
    "rvm_mobilenetv3_human",
    "rvm_resnet50_human"
]

inspyrenet_mdoel_ids = [
    "base",
    "base-nightly"
]

basnet_model_ids = [
    "basnet_refiner",
    "basnet"
]

compared_images = []


def compare():
    compared_images = [(str(img), img.name) for img in Path(file_manager_matting.outputs_dir).rglob("*.png")]
    # 日期超过两天的删除

    return compared_images


def run_rembg(
        input_img: np.ndarray,
        model_id: str,
        background_color,
        post_process_mask,
        alpha_cutout_erode_size,
        alpha_cutout_foreground_threshold,
        alpha_cutout_background_threshold,
        result_type
):
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    # 判断一下，不过这个必须这样因为他会启动判定的，会能换模型的
    # function里面判定没有意义，要写成类判定才有意义，
    session = new_session(model_id)

    img_rgba, pil_mask = remove_matting(
        input_img,
        session=session,
        bgcolor=background_color,
        post_process_mask=post_process_mask,
        alpha_matting_erode_size=alpha_cutout_erode_size,
        alpha_matting_foreground_threshold=alpha_cutout_foreground_threshold,
        alpha_matting_background_threshold=alpha_cutout_background_threshold,
        result_type=result_type)  # 输入什么形式，输出什么形式

    save_name = "_".join([file_manager_matting.savename_prefix, os.path.basename(model_id)]) + ".png"
    save_name = os.path.join(file_manager_matting.outputs_dir, save_name)
    img_rgba.save(save_name)
    return [img_rgba, pil_mask]


def run_ppmatting(
        input,
        ppmatting_model_id,
        background_color,
        result_type,
        morph_op,
        morph_op_factor):
    img_rgba, pil_mask = image_ppmatting(
        input,
        result_type=result_type,
        bg_color=background_color,
        algorithm=ppmatting_model_id,
        morph_op=morph_op,
        morph_op_factor=morph_op_factor)

    save_name = "_".join([file_manager_matting.savename_prefix, os.path.basename(ppmatting_model_id)]) + ".png"
    save_name = os.path.join(file_manager_matting.outputs_dir, save_name)
    img_rgba.save(save_name)
    return [img_rgba, pil_mask]


def run_modnet(
        input,
        result_type,
        background_color
):
    img_rgba, pil_mask = modnet(
        input,
        result_type=result_type,
        bg_color=background_color)

    save_name = "_".join([file_manager_matting.savename_prefix, "modnet"]) + ".png"
    save_name = os.path.join(file_manager_matting.outputs_dir, save_name)
    img_rgba.save(save_name)
    return [img_rgba, pil_mask]


def run_rvm(
        input,
        rvm_model_id,
        result_type,
        background_color
):
    # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    converter = Converter(
        download_models_rvm(rvm_model_id),
        "cpu", "fp32", result_type, background_color)
    img_rgba, pil_mask = converter.convert(input_source=input)

    save_name = "_".join([file_manager_matting.savename_prefix, os.path.basename(rvm_model_id)]) + ".png"
    save_name = os.path.join(file_manager_matting.outputs_dir, save_name)
    img_rgba.save(save_name)
    return [img_rgba, pil_mask]


def run_bshm(
        input,
        bshm_model_id,
        result_type,
        background_color
):
    # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    img_rgba, pil_mask = bshm(input,
                              bshm_model_id,
                              result_type,
                              background_color)
    save_name = "_".join([file_manager_matting.savename_prefix, os.path.basename(bshm_model_id)]) + ".png"
    save_name = os.path.join(file_manager_matting.outputs_dir, save_name)
    img_rgba.save(save_name)
    return [img_rgba, pil_mask]


def run_carvekit(
        input,
        result_type,
        background_color
):
    # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    img_rgba, pil_mask = tracerb7(
        input,
        result_type=result_type,
        bg_color=background_color)

    save_name = "_".join([file_manager_matting.savename_prefix, "tracerb7"]) + ".png"
    save_name = os.path.join(file_manager_matting.outputs_dir, save_name)
    img_rgba.save(save_name)
    return [img_rgba, pil_mask]


def run_inspyrenet(
        input,
        inspyrenet_model_id,
        threshold,
        result_type,
        background_color):
    # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    img_rgba, pil_mask = inspyrenet(
        input,
        inspyrenet_model_id,
        threshold,
        result_type,
        background_color
    )
    save_name = "_".join([file_manager_matting.savename_prefix, os.path.basename(inspyrenet_model_id)]) + ".png"
    save_name = os.path.join(file_manager_matting.outputs_dir, save_name)
    img_rgba.save(save_name)
    return [img_rgba, pil_mask]


def run_basnet(
        input,
        basnet_model_id,
        result_type,
        background_color):
    # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    if basnet_model_id == "basnet_refiner":
        img_rgba, pil_mask = basnet_refiner(
            input,
            result_type,
            background_color)
        save_name = "_".join([file_manager_matting.savename_prefix, os.path.basename(basnet_model_id)]) + ".png"
        save_name = os.path.join(file_manager_matting.outputs_dir, save_name)
        img_rgba.save(save_name)
    else:
        img_rgba, pil_mask = basnet(
            input,
            result_type,
            background_color)
        save_name = "_".join([file_manager_matting.savename_prefix, os.path.basename(basnet_model_id)]) + ".png"
        save_name = os.path.join(file_manager_matting.outputs_dir, save_name)
        img_rgba.save(save_name)

    return [img_rgba, pil_mask]


if __name__ == "__main__":
    # img_np = [cv2.imread(str(img)) for img in Path("/root/autodl-tmp/stable-diffusion-webui/extensions/sd_webui_matting/data").rglob("*.[pj][pn]g")]
    # img_np = [cv2.imread(str(img)) for img in
    #           Path("/root/autodl-tmp/stable-diffusion-webui/extensions/sd_webui_matting/data").rglob("34.jpg")]
    img_list = list(
        Path("/root/autodl-tmp/stable-diffusion-webui/extensions/sd_webui_matting/data").rglob("*.[pj][pn]g"))
    img_list = sorted(
        [image_path for image_path in img_list if "checkpoint" not in image_path.stem],
        key=lambda x: int(x.stem))
    # img_np = [np.array(Image.open(str(img))) for img in img_list]
    img_np = [cv2.imread(str(img)) for img in img_list]
    img_np = [cv2.cvtColor(input, cv2.COLOR_BGR2RGB) for input in img_np]

    #     for model_id in rembg_model_ids:
    #         for img in img_np:
    #             run_rembg(img, model_id, background_color="#FFFFFF", post_process_mask=True, alpha_cutout_erode_size=10,
    #                       alpha_cutout_foreground_threshold=240, alpha_cutout_background_threshold=10,
    #                       result_type="RemoveBG")

    # for model_id in ppmatting_model_ids:
    #     for img in img_np:
    #         run_ppmatting(img, model_id, background_color="#FFFFFF", result_type="RemoveBG", morph_op="Dilate",
    #                       morph_op_factor=0)

    #     for img in img_np:
    #         run_modnet(img, result_type="RemoveBG", background_color="#FFFFFF")

    # for model_id in rvm_model_ids:
    #     for img in img_np:
    #         run_rvm(img, model_id, result_type="RemoveBG", background_color="#FFFFFF")

    for model_id in bshm_model_ids:
        for img in img_np:
            run_bshm(img, model_id, result_type="RemoveBG", background_color="#FFFFFF")

    # for img in img_np:
    #     run_carvekit(img, result_type="RemoveBG", background_color="#FFFFFF")

    # for model_id in inspyrenet_mdoel_ids:
    #     for img in img_np:
    #         run_inspyrenet(img, model_id, threshold=None, result_type="RemoveBG", background_color="#FFFFFF")

    # import pdb;pdb.set_trace()
    # for model_id in basnet_model_ids:
    #     for img in img_np:
    #         run_basnet(img, model_id, result_type="RemoveBG", background_color="#FFFFFF")
