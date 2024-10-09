import os
import re
import PIL
import hashlib
import gradio as gr
from PIL import Image
from typing import Tuple, List
from modules.shared import opts
import modules.scripts as scripts
from modules import script_callbacks
from scripts.fastload import judgeControlnetDataFile, print_info
import modules.generation_parameters_copypaste as parameters_copypaste

picSHA256, allViewData = {}, {}
addEmoji = "➕"
flyEmoji = "✈️"
elemIdFlag = "controlnet_fastload_tab_"
accessLevel = -1

class viewDataWrap:
    def __init__(self, filepathList: list, picDict: dict):
        print("PDebug: Entering     __init__")
        self.filepathList = filepathList
        self.picDict = picDict

class ToolButton(gr.Button, gr.components.FormComponent):
    def __init__(self, **kwargs):
        print("PDebug: Entering     __init__")
        super().__init__(variant="tool", elem_classes=["toolButton"], **kwargs)

    def get_block_name(self):
        print("PDebug: Entering     get_block_name")
        return "button"

def on_ui_tabs() -> list:
    print("PDebug: Entering on_ui_tabs")
    global accessLevel
    from modules.shared import cmd_opts
    isRemote = cmd_opts.share or cmd_opts.ngrok or cmd_opts.listen or cmd_opts.server_name
    accessLevel = int(os.getenv("CONTROLNET_FASTLOAD_FILTER_ACCESS_CONTROL", -1)) if isRemote else 2
    accessToken = str(os.getenv("CONTROLNET_FASTLOAD_FILTER_ACCESS_TOKEN", "")) if accessLevel <= 1 else ""
    tabDebug = True if os.getenv("CONTROLNET_FASTLOAD_DEBUG", "") == "True" else False  # Only for self-test
    viewPathSelectList = ["txt2img", "img2img", "manually"] if accessLevel > 1 else ["txt2img", "img2img"]
    viewPathSelectList = [] if accessLevel == 0 else viewPathSelectList
    print_info(f"PDebug: Load Controlnet Fastload Filter on isRemote={isRemote} and accessLevel={accessLevel}")
    print_info(f"PDebug: You have enabled access token in Controlnet Fastload Filter") if accessToken != "" else None
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        accessTokenInput = gr.Textbox(label="Access Token", visible=(accessToken != ""))
                        accessTokenSubmit = ToolButton(value=flyEmoji, visible=(accessToken != ""))
                        accessTokenRightSHA512 = gr.Textbox(visible=False, interactive=False,
                                                            value=hashlib.sha512(accessToken.encode()).hexdigest())
                        accessTokenSubmit.click(_js="checkAccessToken",
                                                fn=fnaccessTokenSubmit,
                                                inputs=[accessTokenInput, accessTokenRightSHA512],
                                                outputs=[accessTokenInput, accessTokenSubmit])
                    with gr.Row():
                        currentDir = os.path.join(scripts.basedir(), opts.data.get("outdir_txt2img_samples"))
                        if os.path.exists(currentDir) and accessLevel >= 1:
                            path_, fast_way = currentDir, "txt2img"
                        elif (not os.path.exists(currentDir)) and accessLevel >= 2:
                            path_, fast_way = "", "manually"
                        else:
                            path_, fast_way = "", ""
                        with gr.Column(scale=4):
                            viewPath = gr.Textbox(value=path_,
                                                  label="View Path",
                                                  elem_id=f'{elemIdFlag}view_path',
                                                  interactive=False)
                        with gr.Column(scale=1, min_width=180):
                            viewPathSelect = gr.Dropdown(label="View Path Select",
                                                         choices=viewPathSelectList,
                                                         value=fast_way,
                                                         elem_id=f'{elemIdFlag}view_path_select')
                    with gr.Row():
                        with gr.Column(min_width=100):
                            firstPage = gr.Button("First Page")
                        with gr.Column(min_width=100):
                            prevPage = gr.Button("Prev Page")
                        with gr.Column(min_width=100):
                            pageIndex = gr.Number(value=1, label="Page Index")
                        with gr.Column(min_width=100):
                            nextPage = gr.Button("Next Page")
                        with gr.Column(min_width=100):
                            endPage = gr.Button("End Page")
                    lastViewPath = gr.Textbox(visible=False, interactive=False)
                    gallery = gr.Gallery(elem_id="_images_history_gallery", columns=6)
                with gr.Column(scale=2):
                    with gr.Row():
                        # filterKey和filterValue绑定
                        filterKey = gr.Dropdown(label="filter item",
                                                choices=["None"],
                                                value="None",
                                                elem_id=f'{elemIdFlag}view_path_select')
                        filterValueDropDown = gr.Dropdown(label="filter value", choices=[], multiselect=True)
                        filterValueTextbox = gr.Dropdown(label="filter value", value="", visible=False)
                        filterAddAll = ToolButton(value=addEmoji, elem_id=f'{elemIdFlag}filter_button')
                        filterManualSend = ToolButton(value=flyEmoji, elem_id=f'{elemIdFlag}filter_send')
                    with gr.Row():
                        filterAll = gr.Dropdown(label="all filters", choices=[], multiselect=True, interactive=False)
                    with gr.Row():
                        sendControlnetPriority = gr.Dropdown(label="send priority",
                                                             choices=["Controlnet First", "Controlnet Fastload First",
                                                                      "Auto"],
                                                             value="Auto")
                    with gr.Row():
                        diff = gr.HighlightedText(
                            label="ControlNet Info",
                            combine_adjacent=False,
                            show_legend=False,
                            color_map={"include": "green"})
                    with gr.Row():
                        with (gr.Accordion("Other Info", open=False)):
                            otherInfo = gr.HTML()
                            selectPicAddress = gr.Textbox(value="", visible=False, interactive=False)
                            selectPicControlnetAddress = gr.Textbox(value="", visible=False, interactive=False)
                    with gr.Row(equal_height=True):
                        tabDebugBox = gr.Textbox(value="True" if tabDebug else "False", visible=False)
                        with gr.Column(min_width=80):
                            sendTxt2img = gr.Button(value="Send to txt2img", elem_id=f'{elemIdFlag}send_txt2img')
                        with gr.Column(min_width=80):
                            sendImg2img = gr.Button(value="Send to img2img", elem_id=f'{elemIdFlag}send_img2img')
                        with gr.Column(min_width=80):
                            sendControlnetTxt2img = gr.Button(value="Send to Controlnet-txt2img",
                                                              elem_id=f'{elemIdFlag}send_controlnet_txt2img')
                        with gr.Column(min_width=80):
                            sendControlnetImg2img = gr.Button(value="Send to Controlnet-img2img",
                                                              elem_id=f'{elemIdFlag}send_controlnet_img2img')
        parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
            paste_button=sendTxt2img, tabname="txt2img", source_text_component=otherInfo,
            source_image_component=None,
        ))
        parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
            paste_button=sendImg2img, tabname="img2img", source_text_component=otherInfo,
            source_image_component=None,
        ))
        fnLoadPictureInputListBase = [viewPath, viewPathSelect, lastViewPath, filterAll, filterKey, pageIndex]
        fnLoadPictureInputList = lambda obj: fnLoadPictureInputListBase + [obj]
        fnLoadPictureOutputList = [lastViewPath, gallery, filterKey, pageIndex,
                                   diff, otherInfo, filterAll, filterValueDropDown]
        sendControlnetTxt2img.click(fn=None, _js="sendToAny2img",
                                    inputs=[selectPicAddress, selectPicControlnetAddress,
                                            sendControlnetPriority, sendControlnetTxt2img, tabDebugBox])
        sendControlnetImg2img.click(fn=None, _js="sendToAny2img",
                                    inputs=[selectPicAddress, selectPicControlnetAddress,
                                            sendControlnetPriority, sendControlnetImg2img, tabDebugBox])
        viewPath.change(fn=fnViewPathChange,
                        inputs=[viewPath, viewPathSelect, lastViewPath],
                        outputs=fnLoadPictureOutputList)
        viewPathSelect.select(fn=fnViewPathSelect,
                              inputs=viewPathSelect,
                              outputs=viewPath)
        # Bind events for the buttons on the left
        firstPage.click(fn=fnLoadPicture,
                        inputs=fnLoadPictureInputList(firstPage),
                        outputs=fnLoadPictureOutputList)
        prevPage.click(fn=fnLoadPicture,
                       inputs=fnLoadPictureInputList(prevPage),
                       outputs=fnLoadPictureOutputList)
        nextPage.click(fn=fnLoadPicture,
                       inputs=fnLoadPictureInputList(nextPage),
                       outputs=fnLoadPictureOutputList)
        endPage.click(fn=fnLoadPicture,
                      inputs=fnLoadPictureInputList(endPage),
                      outputs=fnLoadPictureOutputList)
        pageIndex.input(fn=fnLoadPicture,
                        inputs=fnLoadPictureInputListBase,
                        outputs=fnLoadPictureOutputList)
        filterManualSend.click(fn=fnLoadPicture,
                               inputs=fnLoadPictureInputListBase,
                               outputs=fnLoadPictureOutputList)
        # Bind filterKey change event
        filterKey.input(fn=fnFilterKeyChange,
                        inputs=[filterKey, filterAll],
                        outputs=[filterValueDropDown, filterValueTextbox, filterAll])
        # Bind "+" button event
        filterAddAll.click(fn=fnFilterAddAll,
                           inputs=[filterKey, filterValueDropDown, filterValueTextbox, filterAll],
                           outputs=[filterAll])
        gallery.select(fn=fnGallerySelect,
                       inputs=[gallery, filterAll],
                       outputs=[diff, otherInfo, selectPicAddress, selectPicControlnetAddress])
        return [(ui_component, "Controlnet Fastload Filter", "controlnet_fastload_filter")]


def fnViewPathChange(viewPath: str, viewPathSelect: str, lastViewPath: str) -> list:
    if viewPathSelect != "manually":
        return fnLoadPicture(viewPath, viewPathSelect, lastViewPath, [], [], 1)
    else:
        return [lastViewPath, [], gr.update(choices=["None"]), 1, [], "", gr.update(value=[]), gr.update(value=[])]


def fnaccessTokenSubmit(accessTokenInput: str, accessTokenRightSHA512: str) -> list:
    global accessLevel
    print("PDebug: Inside fnaccessTokenSubmit")
    if accessTokenInput == str(os.getenv("CONTROLNET_FASTLOAD_FILTER_ACCESS_TOKEN", "")):
        accessLevel = 2
        return [gr.update(visible=False), gr.update(visible=False)]
    else:
        return [gr.update(visible=True), gr.update(visible=True)]

def fnGallerySelect(selectData: gr.SelectData, gallery: list, filterAll: list) -> list:
    print("PDebug: Inside fnGallerySelect")
    selectFile = gallery[selectData.index]['name']  # It is in the temp folder, SHA256 is needed to verify
    with open(selectFile, 'rb') as f:
        originalFile = picSHA256[hashlib.sha256(f.read()).hexdigest()]
    with Image.open(selectFile) as img:
        if "parameters" in img.info:
            infoList = extractControlNet(selectFile, img.info['parameters'], {}, "diff")
        else:
            infoList = []
    result = []
    for info in range(len(infoList)):
        for item in infoList[info]:
            filter_ = f"{item[0]} - {item[1]}"
            if filter_ in filterAll:
                result.append((f"[ControlNet {info}] {filter_}\n", "include"))
            else:
                result.append((f"[ControlNet {info}] {filter_}\n", None))
    returnCNFilePath = judgeControlnetDataFile(originalFile, gallery[selectData.index]['data'])
    print("PDebug: fnGallerySelect completed")
    return [result, img.info['parameters'] if "parameters" in img.info else "",
            gallery[selectData.index]['data'], returnCNFilePath]


def fnViewPathSelect(viewPathSelect: str) -> dict:
    print("PDebug: Inside fnViewPathSelect")
    if viewPathSelect == "txt2img":
        return gr.update(value=os.path.join(scripts.basedir(), opts.data.get("outdir_txt2img_samples")),
                         interactive=False)
    elif viewPathSelect == "img2img":
        return gr.update(value=os.path.join(scripts.basedir(), opts.data.get("outdir_img2img_samples")),
                         interactive=False)
    else:
        return gr.update(value="", interactive=True)


def fnFilterKeyChange(filterKey: str, filterAll: list) -> list:
    print("PDebug: Inside fnFilterKeyChange")
    picDict = {}
    tmpList = [] if filterKey == "None" else [f"{filterKey} - {itm}" for itm in picDict[filterKey].keys()]
    print("PDebug: fnFilterKeyChange completed")
    return [gr.update(visible=True, choices=tmpList, value=[]), gr.update(visible=False), filterAll]


def fnFilterAddAll(filterKey: str, filterValueDropDown: list, filterValueTextbox: str, filterAll: list) -> list:
    print("PDebug: Inside fnFilterAddAll")
    unique_filterAll = set(filterAll)
    if len(filterValueDropDown) > 0:
        unique_filterAll.update(filterValueDropDown)
    print("PDebug: fnFilterAddAll completed")
    return list(unique_filterAll)


def fnLoadPicture(*args) -> list:
    print("PDebug: Inside fnLoadPicture")
    viewPath, viewPathSelect, lastViewPath, filterAll, filterKey, pageIndex = args[:6]
    global allViewData
    if accessLevel <= 0:
        raise gr.Error("You have no permission to use this function")
    if not (os.path.exists(viewPath) and os.path.isdir(viewPath)):
        raise gr.Error(f"ViewPath {viewPath} does not exist or not a folder")
    if viewPath != lastViewPath:
        # Fresh load
        filepathList, picDict = loadPicture(viewPath)
        allViewData[viewPath] = viewDataWrap(filepathList, picDict)
        tmpFilterKey = list(picDict.keys())
        tmpFilterKey.insert(0, "None")
        displayPic, pageIndex_ = loadDisplayPic(*args,
                                                filepathList_=filepathList, pageIndex_=pageIndex)
        calculateSHA256(displayPic)
        # No filter for the first time, display all images
        return [viewPath, gr.update(value=displayPic), gr.update(choices=tmpFilterKey),
                gr.update(value=pageIndex_), [], "", gr.update(value=[]), gr.update(value=[])]
    else:
        # Directly filter filepathList
        filepathList, picDict = (allViewData[viewPath].filepathList, allViewData[viewPath].picDict)
        allSet = set(filepathList)
        for itm in filterAll:
            key, val = itm.split(" - ")
            smallSet = picDict[key][val]
            allSet = allSet.intersection(smallSet)
        displayPic, pageIndex_ = loadDisplayPic(*args, filepathList_=list(allSet), pageIndex_=pageIndex)
        calculateSHA256(displayPic)
        return [viewPath, gr.update(value=displayPic), filterKey,
                gr.update(value=pageIndex_), [], "", gr.update(), gr.update()]


def loadDisplayPic(*args, **kwargs) -> Tuple[List[str], int]:
    print("PDebug: Inside loadDisplayPic")
    pageEnum = {
        "First Page": 0,
        "Prev Page": -1,
        "Next Page": 1,
        "End Page": -1
    }
    argsLenLimit = 6
    perPagePicNum = 36
    pageIndex_, filepathList_ = kwargs["pageIndex_"], kwargs["filepathList_"]
    displayAllPic = [filepathList_[i:i + perPagePicNum] for i in range(0, len(filepathList_), perPagePicNum)]
    # fix pageIndex_
    pageIndex_ = 1 if pageIndex_ is None else pageIndex_
    pageIndex_ = pageIndex_ if 1 <= pageIndex_ <= len(displayAllPic) else 1
    # No page turning operation
    if len(args) <= argsLenLimit:
        displayPic = filepathList_
    # Page turning, first or last page
    elif len(args) > argsLenLimit and (args[argsLenLimit] == "First Page" or args[argsLenLimit] == "End Page"):
        pageIndex_ = 1 if args[argsLenLimit] == "First Page" else len(displayAllPic)
        displayPic = displayAllPic[pageIndex_ - 1]
    # Page turning, previous or next page
    else:
        pageIndex_ = pageIndex_ + pageEnum[args[argsLenLimit]]
        pageIndex_ = pageIndex_ if 1 <= pageIndex_ <= len(displayAllPic) else pageIndex_ - pageEnum[args[argsLenLimit]]
        displayPic = displayAllPic[int(pageIndex_) - 1]
    print("PDebug: loadDisplayPic completed")
    return displayPic, pageIndex_


def loadPicture(filepath: str) -> Tuple[List[str], dict]:
    print("PDebug: Inside loadPicture")
    filepathList_ = []
    picDict_ = {"preprocessor": {}, "model": {}, "weight": {}, "starting/ending": {}, "resize mode": {},
                "pixel perfect": {}, "control mode": {}, "preprocessor params": {}}
    for folderName, subFolders, fileNames in os.walk(filepath):
        for fileName in fileNames:
            fullname = os.path.join(folderName, fileName)
            # Process png_info
            try:
                with Image.open(fullname) as img:
                    if "parameters" in img.info:
                        extractControlNet(fullname, img.info['parameters'], picDict_, "init")
                filepathList_.append(fullname)
            except (PIL.UnidentifiedImageError, IOError, OSError, ValueError):
                pass
    print("PDebug: loadPicture completed")
    return filepathList_, picDict_


def extractControlNet(fullname: str, pngInfo: str, picDict_: dict, mode: str) -> list:
    print(f"PDebug: Inside extractControlNet, mode={mode}")
    res = re.findall(r'ControlNet[^"]+"([^"]+)"', pngInfo)
    pairList = []
    for itm in res:
        pairs = re.findall(r'\s*([^:,]+):\s*(\([^)]+\)|[^,]+)(?:,|$)', itm)
        if mode == "init":
            for key, value in pairs:
                picDict_.setdefault(key, {}).setdefault(value, set()).add(fullname)
        elif mode == "diff":
            pairList.append(pairs)
        else:
            pass
    print(f"PDebug: extractControlNet completed, mode={mode}")
    return pairList if mode == "diff" else None


def calculateSHA256(fileList: list) -> None:
    print("PDebug: Inside calculateSHA256")
    global picSHA256
    for file in fileList:
        with open(file, 'rb') as f:
            picSHA256[hashlib.sha256(f.read()).hexdigest()] = file
    print("PDebug: calculateSHA256 completed")


script_callbacks.on_ui_tabs(on_ui_tabs)

