# api
# import sys
# sys.path.append("/root/autodl-tmp/stable-diffusion-webui")

import os
import string
import random
from modules import shared
from datetime import datetime
from pathlib import Path


def get_webui_setting(key, default):
    value = shared.opts.data.get(key, default)

    if not isinstance(value, type(default)):
        value = default
    return value


# 生成随机字符串
def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


class FileManager:
    def __init__(self) -> None:
        self.update_outputs_dir()

    def update_outputs_dir(self) -> None:
        config_save_folder = get_webui_setting("matting_save_folder", "matting")
        self._outputs_dir = os.path.join(shared.data_path, "outputs", config_save_folder,
                                         datetime.now().strftime("%Y-%m-%d"))

    @property
    def outputs_dir(self) -> str:
        self.update_outputs_dir()
        if not os.path.isdir(self._outputs_dir):
            os.makedirs(self._outputs_dir, exist_ok=True)
        return self._outputs_dir

    @property
    def savename_prefix(self) -> str:
        config_save_folder = get_webui_setting("matting_save_folder", "matting")
        self.folder_is_webui = False if config_save_folder == "matting" else True
        basename = "matting-" if self.folder_is_webui else ""
        return basename + datetime.now().strftime("%Y%m%d-%H%M%S") + f"{generate_random_string(4)}"


file_manager = FileManager()
