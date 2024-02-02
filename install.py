import launch

if not launch.is_installed("gradio"):
    try:
        launch.run_pip("install gradio==3.32", "requirements for gradio")
    except Exception:
        print("Can't install gradio. Please follow the readme to install manually")

if not launch.is_installed("paddleseg"):
    try:
        launch.run_pip("install paddleseg==2.7.0", "requirements for gradio")
    except Exception:
        print("Can't install paddleseg. Please follow the readme to install manually")

if not launch.is_installed("paddlepaddle_gpu"):
    try:
        launch.run_pip("install paddlepaddle_gpu", "requirements for paddlepaddle_gpu")
    except Exception:
        print("Can't install paddlepaddle_gpu. Please follow the readme to install manually")

if not launch.is_installed("carvekit_colab"):
    try:
        launch.run_pip("install carvekit_colab", "requirements for carvekit_colab")
    except Exception:
        print("Can't install carvekit_colab. Please follow the readme to install manually")

if not launch.is_installed("modelscope"):
    try:
        launch.run_pip("install modelscope", "requirements for modelscope")
    except Exception:
        print("Can't install modelscope. Please follow the readme to install manually")

if not launch.is_installed("tensorflow"):
    try:
        launch.run_pip("install tensorflow", "requirements for tensorflow")
    except Exception:
        print("Can't install tensorflow. Please follow the readme to install manually")

if not launch.is_installed("transparent-background"):
    try:
        launch.run_pip("install transparent-background", "requirements for transparent-background")
    except Exception:
        print("Can't install transparent-background. Please follow the readme to install manually")

if not launch.is_installed("av"):
    try:
        launch.run_pip("install av", "requirements for av")
    except Exception:
        print("Can't install av. Please follow the readme to install manually")

if not launch.is_installed("pims"):
    try:
        launch.run_pip("install pims", "requirements for pims")
    except Exception:
        print("Can't install pims. Please follow the readme to install manually")

if not launch.is_installed("onnxruntime_gpu"):
    try:
        launch.run_pip("install onnxruntime_gpu", "requirements for onnxruntime_gpu")
    except Exception:
        print("Can't install onnxruntime_gpu. Please follow the readme to install manually")

if not launch.is_installed("pooch"):
    try:
        launch.run_pip("install pooch", "requirements for pooch")
    except Exception:
        print("Can't install pooch. Please follow the readme to install manually")

if not launch.is_installed("pymatting"):
    try:
        launch.run_pip("install pymatting", "requirements for pymatting")
    except Exception:
        print("Can't install pymatting. Please follow the readme to install manually")

# if not launch.is_installed("mmcv"):
#     try:
#         launch.run_pip("install mmcv", "requirements for mmcv")
#     except Exception:
#         print("Can't install mmcv. Please follow the readme to install manually")
