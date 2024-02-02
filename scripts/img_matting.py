import os
import cv2
import numpy as np
import gradio as gr
from pathlib import Path

from modules import script_callbacks
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
    # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
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
    if basnet_model_id == "basenet_refiner":
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


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as matting_interface:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input = gr.Image(label="Input", type="numpy")

                with gr.Tab("rembg", elem_id='rembg_tab'):
                    with gr.Row():
                        with gr.Column():
                            rembg_model_id = gr.Dropdown(label="Model ID", elem_id="model_id", choices=rembg_model_ids,
                                                         value=rembg_model_ids[0], show_label=True)
                            with gr.Row():
                                post_process_mask = gr.Checkbox(label="Post process mask", elem_id="post_process_mask",
                                                                show_label=True, interactive=True)
                            with gr.Row(variant="compact"):
                                result_type = gr.Radio(label="Mode", show_label=True,
                                                       choices=["RemoveBG", "ReplaceBG"], value="RemoveBG")
                                background_color = gr.ColorPicker(label="Bg color", value="#FFFFFF")

                        with gr.Column():
                            alpha_cutout_erode_size = gr.Slider(label="Erode size", minimum=0, maximum=255, step=1,
                                                                value=10)
                            alpha_cutout_foreground_threshold = gr.Slider(label="Foreground threshold", minimum=0,
                                                                          maximum=255, step=1, value=240)
                            alpha_cutout_background_threshold = gr.Slider(label="Background threshold", minimum=0,
                                                                          maximum=255, step=1, value=10)

                    with gr.Row():
                        rembg_btn = gr.Button("Run Rembg", elem_id="rembg_btn", variant="primary")

                with gr.Tab("ppmatting", elem_id="ppmatting_tab"):
                    with gr.Row():
                        with gr.Column():
                            ppmatting_model_id = gr.Dropdown(label="Model ID", elem_id='model_id',
                                                             choices=ppmatting_model_ids,
                                                             value=ppmatting_model_ids[0], show_label=True)
                            with gr.Row(variant="compact"):
                                result_type_pp = gr.Radio(label="Mode", show_label=True,
                                                          choices=["RemoveBG", "ReplaceBG"], value="RemoveBG")
                                background_color_pp = gr.ColorPicker(label="Bg color", value="#FFFFFF")

                        with gr.Column():
                            morph_op = gr.Radio(label="Post-process", show_label=True,
                                                choices=["Dilate", "Erode", "None"], value="Dilate")

                            morph_op_factor = gr.Slider(label="Factor", show_label=True,
                                                        minimum=0, maximum=20, value=0, step=1)

                    with gr.Row():
                        ppmatting_btn = gr.Button("Run PPMatting", elem_id="ppmatting_btn", variant="primary")

                with gr.Tab("modnet", elem_id='modnet_tab'):
                    with gr.Row():
                        result_type_modnet = gr.Radio(label="Mode", show_label=True,
                                                      choices=["RemoveBG", "ReplaceBG"], value="RemoveBG")
                        background_color_modnet = gr.ColorPicker(label="Bg color", value="#FFFFFF")

                    with gr.Row():
                        modnet_btn = gr.Button("Run Modnet", elem_id="modnet_btn", variant="primary")

                with gr.Tab("bshm", elem_id="bshm_tab"):
                    with gr.Row():
                        bshm_model_id = gr.Dropdown(label="Model ID", elem_id='model_id',
                                                    choices=bshm_model_ids,
                                                    value=bshm_model_ids[0], show_label=True)
                        with gr.Column():
                            result_type_bshm = gr.Radio(label="Mode", show_label=True,
                                                        choices=["RemoveBG", "ReplaceBG"], value="RemoveBG")
                            background_color_bshm = gr.ColorPicker(label="Bg color", value="#FFFFFF")

                    with gr.Row():
                        bshm_btn = gr.Button("Run BSHM", elem_id="bshm_btn", variant="primary")

                with gr.Tab("rvm", elem_id="rvm_tab"):
                    with gr.Row():
                        rvm_model_id = gr.Dropdown(label="Model ID", elem_id='model_id',
                                                   choices=rvm_model_ids,
                                                   value=rvm_model_ids[0], show_label=True)
                        with gr.Column():
                            result_type_rvm = gr.Radio(label="Mode", show_label=True,
                                                       choices=["RemoveBG", "ReplaceBG"], value="RemoveBG")
                            background_color_rvm = gr.ColorPicker(label="Bg color", value="#FFFFFF")

                    with gr.Row():
                        rvm_btn = gr.Button("Run RVM", elem_id="rvm_btn", variant="primary")

                with gr.Tab("tracrb7", elem_id='tracerb7_tab'):
                    with gr.Row():
                        result_type_tracerb7 = gr.Radio(label="Mode", show_label=True,
                                                        choices=["RemoveBG", "ReplaceBG"], value="RemoveBG")
                        background_color_tracerb7 = gr.ColorPicker(label="Bg color", value="#FFFFFF")

                    with gr.Row():
                        tracerb7_btn = gr.Button("Run TracerB7", elem_id="tracerb7_btn", variant="primary")

                with gr.Tab("inspyrenet", elem_id="inspyrenet_tab"):
                    with gr.Row():
                        with gr.Column():
                            inspyrenet_model_id = gr.Dropdown(label="Model ID", elem_id='model_id',
                                                              choices=inspyrenet_mdoel_ids,
                                                              value=inspyrenet_mdoel_ids[0], show_label=True)
                            threshold = gr.Slider(label="Threashold", minimum=0, maximum=1, step=0.1,
                                                  value=0)
                        with gr.Column():
                            result_type_inspyrenet = gr.Radio(label="Mode", show_label=True,
                                                              choices=["RemoveBG", "ReplaceBG"], value="RemoveBG")
                            background_color_inspyrenet = gr.ColorPicker(label="Bg color", value="#FFFFFF")

                    with gr.Row():
                        inspyrenet_btn = gr.Button("Run Inspyrenet", elem_id="inspyrenet_btn", variant="primary")

                with gr.Tab("basnet", elem_id="basnet_tab"):
                    with gr.Row():
                        basnet_model_id = gr.Dropdown(label="Model ID", elem_id='model_id',
                                                      choices=basnet_model_ids,
                                                      value=basnet_model_ids[0], show_label=True)
                        with gr.Column():
                            result_type_basnet = gr.Radio(label="Mode", show_label=True,
                                                          choices=["RemoveBG", "ReplaceBG"], value="RemoveBG")
                            background_color_basnet = gr.ColorPicker(label="Bg color", value="#FFFFFF")

                    with gr.Row():
                        basnet_btn = gr.Button("Run Basnet", elem_id="basnet_btn", variant="primary")

            with gr.Column():
                out_gallery_kwargs = dict(columns=2, object_fit="contain", preview=True)
                output = gr.Gallery(label="output", elem_id="output", show_label=True).style(
                    **out_gallery_kwargs)

                send_btn = gr.Button("Send to Compare", elem_id="send_btn", variant="primary")

        with gr.Row():
            compared = gr.Gallery(
                label="Compared images", show_label=True, elem_id="gallery").style(columns=5, rows=2,
                                                                                   object_fit="contain", height="auto")
            # , columns=[5], rows=[2], object_fit="contain", height="auto")

        rembg_btn.click(run_rembg,
                        inputs=[input,
                                rembg_model_id,
                                background_color,
                                post_process_mask,
                                alpha_cutout_erode_size,
                                alpha_cutout_foreground_threshold,
                                alpha_cutout_background_threshold,
                                result_type],
                        outputs=[output])

        ppmatting_btn.click(run_ppmatting,
                            inputs=[input,
                                    ppmatting_model_id,
                                    background_color_pp,
                                    result_type_pp,
                                    morph_op,
                                    morph_op_factor],
                            outputs=[output])

        modnet_btn.click(run_modnet,
                         inputs=[input,
                                 result_type_modnet,
                                 background_color_modnet],
                         outputs=[output])

        rvm_btn.click(run_rvm,
                      inputs=[input,
                              rvm_model_id,
                              result_type_rvm,
                              background_color_rvm],
                      outputs=[output])

        bshm_btn.click(run_bshm,
                       inputs=[input,
                               bshm_model_id,
                               result_type_bshm,
                               background_color_bshm],
                       outputs=[output])

        tracerb7_btn.click(run_carvekit,
                           inputs=[input,
                                   result_type_tracerb7,
                                   background_color_tracerb7],
                           outputs=[output])

        inspyrenet_btn.click(run_inspyrenet,
                             inputs=[input,
                                     inspyrenet_model_id,
                                     threshold,
                                     result_type_inspyrenet,
                                     background_color_inspyrenet],
                             outputs=[output])

        basnet_btn.click(run_basnet,
                         inputs=[input,
                                 basnet_model_id,
                                 result_type_basnet,
                                 background_color_basnet],
                         outputs=[output])

        # 直接存放文件夹，去文件夹中找吧
        send_btn.click(compare, inputs=None, outputs=compared)

    return [(matting_interface, "Matting", "Matting")]


script_callbacks.on_ui_tabs(on_ui_tabs)
