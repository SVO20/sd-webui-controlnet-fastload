import os
import re
import gzip
import pickle
import base64
import importlib
import gradio as gr
import numpy as np
from PIL import Image
from typing import Optional, List
from datetime import datetime
from gradio import Checkbox, Dropdown, File, Textbox, Button, Gallery, JSON
import modules.scripts as scripts
from modules import script_callbacks
from modules.script_callbacks import ImageSaveParams
from modules.shared import opts, cmd_opts
from modules.images import read_info_from_image
from modules.processing import process_images, Processed
import modules.generation_parameters_copypaste as parameters_copypaste

save_flag = False
controlNetList = []
save_filetype = ""
overwrite_flag = ""
start_marker = b'###START_OF_CONTROLNET_FASTLOAD###'
end_marker = b'###END_OF_CONTROLNET_FASTLOAD###'
current_timestamp = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
print_err = lambda msg: print(f'{current_timestamp()} - ControlNetFastload - \033[91mERROR\033[0m - {msg}')
print_warn = lambda msg: print(f'{current_timestamp()} - ControlNetFastload - \033[93mWARNING\033[0m - {msg}')
print_info = lambda msg: print(f'{current_timestamp()} - ControlNetFastload - \033[92mINFO\033[0m - {msg}')
print_debug = lambda msg: print(f'{current_timestamp()} - PDebug - {msg}')

class ControlNetFastLoad(scripts.Script):
    def __init__(self):
        print_debug("Entering __init__")
        pass

    def title(self) -> str:
        print_debug("Entering title")
        return "ControlNet Fastload"

    def show(self, is_img2img: bool) -> bool:
        print_debug("Entering show")
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> list[Checkbox | Dropdown | File | Textbox | Button | Gallery | JSON]:
        print_debug("Entering ui")
        ui_list = []
        with (gr.Accordion("ControlNet Fastload v1.2.1", open=False, elem_id=self.elem_id(""))):
            with gr.Tab("Load data from file"):
                with gr.Row():
                    enabled = gr.Checkbox(value=False, label="Enable", elem_id=self.elem_id("cnfl_enabled"))
                    mode = gr.Dropdown(["Load Only", "Save Only", "Load & Save"], label="Mode", value="Load Only", elem_id=self.elem_id("cnfl_mode"))
                    ui_list.extend([enabled, mode])
                with gr.Row():
                    png_other_info = gr.Textbox(visible=False, elem_id="pnginfo_generation_info")
                    uploadFile = gr.File(type="binary", label="Upload Image or .cni file", file_types=["image", ".cni"], elem_id=self.elem_id("cnfl_uploadImage"))
                    uploadFile.upload(fn=uploadFileListen, inputs=[uploadFile, enabled], outputs=png_other_info)
                    ui_list.extend([uploadFile, png_other_info])
                with gr.Row():
                    visible_ = opts.data.get("isEnabledManualSend")
                    visible_ = False if visible_ is None else visible_
                    send_to_txt2img = gr.Button(value="Send to txt2img", elem_id=self.elem_id("send_to_txt2img"), visible=((not is_img2img) and visible_))
                    send_to_img2img = gr.Button(value="Send to img2img", elem_id=self.elem_id("send_to_img2img"), visible=(is_img2img and visible_))
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=send_to_txt2img, tabname="txt2img", source_text_component=png_other_info, source_image_component=None))
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=send_to_img2img, tabname="img2img", source_text_component=png_other_info, source_image_component=None))
                    ui_list.extend([send_to_txt2img, send_to_img2img])
            with gr.Tab("View saved data"):
                with gr.Row():
                    execute_view_tab = gr.Button(value="Execute", elem_id=self.elem_id("cnfl_execute_view_tab"))
                with gr.Row():
                    uploadFile_view_tab = gr.File(type="binary", label="Upload Image or .cni file", file_types=["image", ".cni"], elem_id=self.elem_id("cnfl_uploadImage_view_tab"))
                with gr.Row():
                    img_view_tab = gr.Gallery(type="file", label="Image data view", elem_id=self.elem_id("cnfl_img_view_tab"), rows=2, columns=2, allow_preview=True, show_download_button=True, object_fit="contain", show_label=True)
                with gr.Row():
                    text_view_tab = gr.Json(label="Text data view", elem_id=self.elem_id("cnfl_text_view_tab"))
                ui_list.extend([execute_view_tab, uploadFile_view_tab, img_view_tab, text_view_tab])
                execute_view_tab.click(fn=viewSaveDataExecute, inputs=[uploadFile_view_tab], outputs=[img_view_tab, text_view_tab])
        return ui_list

    def before_process(self, p, *args) -> None:
        print_debug("Entering before_process")
        api_module = importlib.import_module('extensions.sd-webui-controlnet-fastload.scripts.api')
        api_package = getattr(api_module, "api_package")
        if type(args[0]) is not bool:
            enabled, mode, uploadFile = True, args[0]['mode'], args[0]['filepath']
            saveControlnet, overwritePriority = "", args[0]['overwritePriority']
            api_package.api_instance.enabled = True
            api_package.api_instance.drawId[id(p)] = []
            api_package.api_instance.info()
        else:
            enabled, mode, uploadFile = args[:3]
            saveControlnet, overwritePriority = opts.saveControlnet, opts.overwritePriority
        if enabled:
            try:
                global controlNetList
                break_load = False
                controlNetModule = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
                controlNetList = controlNetModule.get_all_units_in_processing(p)
                controlNetListOriLen = len(controlNetList)
                controlNetListIsEmpty = not (any(itm.enabled for itm in controlNetList))
                if uploadFile is None and (mode == "Load Only" or mode == "Load & Save"):
                    print_warn("Script received no input; the loading process will be skipped.")
                    break_load = True
            except ImportError:
                print_debug("Exception occurred")
                print_warn("ControlNet not found; the script will not work.")
                return
            if (mode == "Load Only" or mode == "Load & Save") and not break_load:
                load_file_name_ = uploadFile if isinstance(uploadFile, str) else uploadFile.name
                if controlNetListIsEmpty:
                    controlNetList = loadFromFile(load_file_name_)
                else:
                    if overwritePriority == "ControlNet Plugin First":
                        print_warn("The plugin is not empty and has priority; the script will not work.")
                    else:
                        print_warn("The plugin is not empty, but the script has priority; it will overwrite the existing ControlNet plugin data.")
                        controlNetList = loadFromFile(load_file_name_)
                if len(controlNetList) > controlNetListOriLen:
                    print_warn("The ControlNet count in the file exceeds the current setting; this might cause an error.")
                controlNetModule.update_cn_script_in_processing(p, controlNetList)
            if mode == "Save Only" or mode == "Load & Save":
                global save_flag, save_filetype
                save_flag = True
                save_filetype = saveControlnet
                if api_package.api_instance.enabled:
                    api_package.api_instance.drawId[id(p)] = controlNetList

    def postprocess_image(self, p, pp, *args):
        print_debug("Entering postprocess_image")
        if type(args[0]) is not bool and args[0]['mode'] != "Load Only":
            p.extra_generation_params['ControlNetID'] = id(p)

def uploadFileListen(pic: gr.File, enabled: bool) -> str:
    print_debug("Entering uploadFileListen")
    if not pic:
        return ""
    if isinstance(pic, gr.File):
        filetype_is_cni = lambda filename: os.path.splitext(pic.name)[1] == '.cni'
        if filetype_is_cni(pic.name) or not enabled:
            return ""
    else:
        print_warn("The uploaded file is not a valid gr.File object.")
        return ""

    fileInPil = Image.open(pic.name)
    gen_info, items = read_info_from_image(fileInPil)
    print(gen_info)
    return gen_info

def judgeControlnetDataFile(filepath: str, filepathWeb: str) -> str:
    print_debug("Entering judgeControlnetDataFile")
    urlStart = re.search(r'^(.*?)/file=', filepathWeb).group(1)
    cnList = loadFromFile(filepath, False)
    cniFilePath = filepath[:-4] + ".cni"
    if len(cnList) > 0:
        return filepathWeb
    elif len(cnList) == 0 and os.path.exists(cniFilePath):
        cnList = loadFromFile(cniFilePath, False)
        return f"{urlStart}/file={filepath[:-4]}.cni" if len(cnList) > 0 else ""
    else:
        return ""

def viewSaveDataExecute(file: gr.File or str) -> tuple:
    """
    View saved ControlNet data from the image/.cni file
    :param file: Uploaded image/file, passed in as wrapped gr.File/str format
    :return: tuple: (list, list) Refer to the UI rendering part for details; this tuple is fed to two UI components
    """
    print_debug("Entering viewSaveDataExecute")
    try:
        if file is None:
            print_warn("You did not upload an image or file.")
            return [], {"Error": "You did not upload an image or file."}
        file_name_ = file if isinstance(file, str) else file.name
        tmpControlNetList = loadFromFile(file_name_)
        previewPicture = []
        previewInformation = []
        loop_count = 0
        for itm in tmpControlNetList:
            tmp = itm if isinstance(itm, dict) else vars(itm)
            if "image" in tmp and tmp["image"] is not None:
                if isinstance(tmp["image"], np.ndarray):
                    image_arrays = [(tmp["image"], f"Controlnet - {loop_count}")]
                else:
                    image_arrays = [(img_array, f"Controlnet - {loop_count}") for img_array in tmp["image"].values()]
                previewPicture.extend(image_arrays)
                tmp.pop("image")
            previewInformation.append(tmp)
            loop_count += 1
        return previewPicture, previewInformation
    except Exception as e:
        print_err(e)
        return [], {"Error": "An unknown error occurred, see the console for details"}

def addToPicture(image: str, datalist: list, imageType: str) -> bytes | None:
    """
    Serialize and store ControlNetList into an image after compressing with gzip
    :param image: Image path or base64-encoded string
    :param datalist: ControlNetList
    :param imageType: "filepath" / "base64"
    """
    print_debug("Entering addToPicture")
    if imageType == "filepath" and (not os.path.exists(image)):
        print_err(f"File {image} does not exist.")
        return
    serialized_data = gzip.compress(pickle.dumps(datalist))
    if imageType == "filepath":
        with open(image, 'rb') as img_file:
            image_data = img_file.read()
    else:
        image_data = base64.b64decode(image)
    combined_data = image_data + start_marker + serialized_data + end_marker
    if imageType == "filepath":
        with open(image, 'wb') as img_file:
            img_file.write(combined_data)
    else:
        return base64.b64encode(combined_data)

def loadFromFile(filepath: str, enableWarn: Optional[bool] = None) -> list:
    """
    Load ControlNetList from an image file
    :param filepath: Image file path
    :param enableWarn: Whether to enable warning messages
    """
    print_debug("Entering loadFromFile")
    if not os.path.exists(filepath):
        if enableWarn is None:
            print_err(f"File {filepath} does not exist.")
        return [{"Error": f"File {filepath} does not exist."}]
    with open(filepath, 'rb') as fp:
        readyLoadData = fp.read()
    start_idx = readyLoadData.find(start_marker) + len(start_marker)
    end_idx = readyLoadData.find(end_marker)
    try:
        embedded_data = gzip.decompress(readyLoadData[start_idx:end_idx])
        readyLoadList = pickle.loads(embedded_data)
        return readyLoadList
    except gzip.BadGzipFile:
        if enableWarn is None:
            print_err(f"{filepath} does not contain valid Controlnet Fastload data.")
        return [{"Error": f"{filepath} does not contain valid Controlnet Fastload data."}]
    except Exception as e:
        if enableWarn is None:
            print_err(f"Error while loading Controlnet Fastload data from the image: {e}")
        return [{"Error": f"Error while loading Controlnet Fastload data from the image: {e}"}]

def afterSavePicture(img_save_param: ImageSaveParams) -> None:
    """
    Hook function to save ControlNetList into an image after it has been saved
    :param img_save_param: Refer to script_callbacks.py
    """
    print_debug("Entering afterSavePicture")
    if save_flag:
        filepath = os.path.join(os.getcwd(), img_save_param.filename)
        filepath_pure, _ = os.path.splitext(filepath)
        if save_filetype == "Embed photo" or save_filetype == "Both":
            addToPicture(filepath, controlNetList, "filepath")
        if save_filetype == "Extra .cni file" or save_filetype == "Both":
            with open(filepath_pure + ".cni", 'wb'):
                pass
            addToPicture(filepath_pure + ".cni", controlNetList, "filepath")
        print_info(f"ControlNet data saved to {filepath}")

script_callbacks.on_image_saved(afterSavePicture)