import numpy as np
import gradio as gr

rembg_model_ids = [
    "lama",
    "ldm",
    "zits",
    "mat",
    "fcf",
    "manga",
]

ppmatting_model_ids = [
    ""
]

modnet_model_ids = [""]

bshm_model_ids = [""]

rvm_model_ids = [""]

basnet_model_ids = [""]


def sepia(inputs, model_id):
    # input_img = inputs["image"]
    # input_mask = inputs["mask"]
    input_img = inputs
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img


compared_images = []


def compare():
    return compared_images


with gr.Blocks(analytics_enabled=False) as lama_interface:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input = gr.Image(label="Input", type="numpy")

            with gr.Tab("rembg", elem_id='rembg_tab'):
                with gr.Row():
                    with gr.Column():
                        rembg_model_id = gr.Dropdown(label="Model ID", elem_id="model_id", choices=rembg_model_ids,
                                                     value=rembg_model_ids[0], show_label=True)
                        background_color = gr.ColorPicker(label="Bg color", value="#FFFFFF")
                        post_process_mask = gr.Checkbox(label="Post process mask", elem_id="post_process_mask",
                                                        show_label=True, interactive=True)

                    with gr.Column():
                        alpha_cutout_erode_size = gr.Slider(label="Erode size", minimum=0, maximum=255, step=1,
                                                            value=0)
                        alpha_cutout_foreground_threshold = gr.Slider(label="Foreground threshold", minimum=0,
                                                                      maximum=255, step=1, value=0)
                        alpha_cutout_background_threshold = gr.Slider(label="Background threshold", minimum=0,
                                                                      maximum=255, step=1, value=0)

                with gr.Row():
                    rembg_btn = gr.Button("Run Rembg", elem_id="rembg_btn", variant="primary")

            with gr.Tab("ppmatting", elem_id="ppmatting_tab"):
                with gr.Row():
                    ppmatting_model_id = gr.Dropdown(label="Model ID", elem_id='model_id', choices=ppmatting_model_ids,
                                                     value=ppmatting_model_ids[0], show_label=True)
                    with gr.Column():
                        ppmatting_btn = gr.Button("Run PPMatting", elem_id="ppmatting_btn", variant="primary")

            with gr.Tab("modnet", elem_id='modnet_tab'):
                with gr.Row():
                    modnet_model_id = gr.Dropdown(label="Model ID", elem_id="model_id", choices=modnet_model_ids,
                                                  value=modnet_model_ids[0], show_label=True)
                    with gr.Column():
                        modnet_btn = gr.Button("Run Modnet", elem_id="modnet_btn", variant="primary")

            with gr.Tab("bshm", elem_id="bshm_tab"):
                with gr.Row():
                    bshm_model_id = gr.Dropdown(label="Model ID", elem_id='model_id', choices=bshm_model_ids,
                                                value=bshm_model_ids[0], show_label=True)
                    with gr.Column():
                        bshm_btn = gr.Button("Run BSHM", elem_id="bshm_btn", variant="primary")

            with gr.Tab("rvmv2", elem_id="rvmv2_tab"):
                with gr.Row():
                    rvm_model_id = gr.Dropdown(label="Model ID", elem_id='model_id', choices=rvm_model_ids,
                                               value=rvm_model_ids[0], show_label=True)
                    with gr.Column():
                        rvm_btn = gr.Button("Run RVM", elem_id="rvm_btn", variant="primary")

            with gr.Tab("basnet", elem_id="basnet_tab"):
                with gr.Row():
                    basnet_model_id = gr.Dropdown(label="Model ID", elem_id='model_id', choices=basnet_model_ids,
                                                  value=basnet_model_ids[0], show_label=True)
                    with gr.Column():
                        basnet_btn = gr.Button("Run BASNet", elem_id="basnet_btn", variant="primary")

        with gr.Column():
            out_gallery_kwargs = dict(columns=2, object_fit="contain", preview=True)
            output = gr.Gallery(label="output", elem_id="output", show_label=True).style(
                **out_gallery_kwargs)

            send_btn = gr.Button("Send to Compare", elem_id="send_btn", variant="primary")

    with gr.Row():
        compared = gr.Gallery(
            label="Compared images", show_label=True, elem_id="gallery"
            , columns=[2], rows=[2], object_fit="contain", height="auto")

    rembg_btn.click(sepia, inputs=[input, rembg_model_id, background_color, post_process_mask, alpha_cutout_erode_size,
                                   alpha_cutout_foreground_threshold, alpha_cutout_background_threshold],
                    outputs=[output])

    send_btn.click(compare, inputs=None, outputs=compared)

lama_interface.launch()
