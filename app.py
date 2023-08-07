#-*- coding: utf-8 -*-
import gradio as gr
from modules.whisper_Inference import WhisperInference
from modules.nllb_inference import NLLBInference
import os
from ui.htmls import *
from modules.youtube_manager import get_ytmetas
import argparse

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import base64
import time

import threading
from fastapi import FastAPI, Request
import uvicorn
import glob
from do_callback import do_callback

class TranscribeFile(BaseModel):
    input_file : Union[str, None] = None
    input_data : Union[str, None] = None
    dd_model : Union[str, None] = None
    dd_lang : Union[str, None] = None
    dd_subformat : Union[str, None] = None
    cb_translate : Union[str, None] = None
    uuid :  Union[str, None] = None
    callbackurl : Union[str, None] = None
    userid : Union[str, None] = None
    userno : Union[str, None] = None
    itemno : Union[str, None] = None
    orig_file : Union[str, None] = None

from datetime import datetime

import uuid

app = FastAPI()

class PredictionRequest(BaseModel):
    input: str

class PredictionResponse(BaseModel):
    output: str

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Add your prediction logic here
    output = "Your prediction result"

    return PredictionResponse(output=output)

# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--share', type=bool, default=False, nargs='?', const=True,
                    help='Share value')
args = parser.parse_args()

def open_folder(folder_path):
    if os.path.exists(folder_path):
        os.system(f"start {folder_path}")
    else:
        print(f"The folder {folder_path} does not exist.")


def on_change_models(model_size):
    translatable_model = ["large", "large-v1", "large-v2"]
    if model_size not in translatable_model:
        return gr.Checkbox.update(visible=False, value=False, interactive=False)
    else:
        return gr.Checkbox.update(visible=True, value=False, label="Translate to English?", interactive=True)

whisper_inf = WhisperInference()
nllb_inf = NLLBInference()
block = gr.Blocks(css=CSS).queue(api_open=False)

with block:
    with gr.Row():
        with gr.Column():
            gr.Markdown(MARKDOWN, elem_id="md_project")
    with gr.Tabs():
        with gr.TabItem("File"):  # tab1
            with gr.Row():
                input_file = gr.Files(type="file", label="Upload File here")
            with gr.Row():
                dd_model = gr.Dropdown(choices=whisper_inf.available_models, value="large-v2", label="Model")
                dd_lang = gr.Dropdown(choices=["Automatic Detection"] + whisper_inf.available_langs,
                                      value="Automatic Detection", label="Language")
                dd_subformat = gr.Dropdown(["SRT", "WebVTT"], value="SRT", label="Subtitle Format")
            with gr.Row():
                cb_translate = gr.Checkbox(value=False, label="Translate to English?", interactive=True)
            with gr.Row():
                btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
            with gr.Row():
                tb_indicator = gr.Textbox(label="Output")
                btn_openfolder = gr.Button('?��').style(full_width=False)

            btn_run.click(fn=whisper_inf.transcribe_file,
                          inputs=[input_file, dd_model, dd_lang, dd_subformat, cb_translate], outputs=[tb_indicator])
            btn_openfolder.click(fn=lambda: open_folder("outputs"), inputs=None, outputs=None)
            dd_model.change(fn=on_change_models, inputs=[dd_model], outputs=[cb_translate])

        with gr.TabItem("Youtube"):  # tab2
            with gr.Row():
                tb_youtubelink = gr.Textbox(label="Youtube Link")
            with gr.Row().style(equal_height=True):
                with gr.Column():
                    img_thumbnail = gr.Image(label="Youtube Thumbnail")
                with gr.Column():
                    tb_title = gr.Label(label="Youtube Title")
                    tb_description = gr.Textbox(label="Youtube Description", max_lines=15)
            with gr.Row():
                dd_model = gr.Dropdown(choices=whisper_inf.available_models, value="large-v2", label="Model")
                dd_lang = gr.Dropdown(choices=["Automatic Detection"] + whisper_inf.available_langs,
                                      value="Automatic Detection", label="Language")
                dd_subformat = gr.Dropdown(choices=["SRT", "WebVTT"], value="SRT", label="Subtitle Format")
            with gr.Row():
                cb_translate = gr.Checkbox(value=False, label="Translate to English?", interactive=True)
            with gr.Row():
                btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
            with gr.Row():
                tb_indicator = gr.Textbox(label="Output")
                btn_openfolder = gr.Button('?��').style(full_width=False)

            btn_run.click(fn=whisper_inf.transcribe_youtube,
                          inputs=[tb_youtubelink, dd_model, dd_lang, dd_subformat, cb_translate],
                          outputs=[tb_indicator])
            tb_youtubelink.change(get_ytmetas, inputs=[tb_youtubelink],
                                  outputs=[img_thumbnail, tb_title, tb_description])
            btn_openfolder.click(fn=lambda: open_folder("outputs"), inputs=None, outputs=None)
            dd_model.change(fn=on_change_models, inputs=[dd_model], outputs=[cb_translate])

        with gr.TabItem("Mic"):  # tab3
            with gr.Row():
                mic_input = gr.Microphone(label="Record with Mic", type="filepath", interactive=True)
            with gr.Row():
                dd_model = gr.Dropdown(choices=whisper_inf.available_models, value="large-v2", label="Model")
                dd_lang = gr.Dropdown(choices=["Automatic Detection"] + whisper_inf.available_langs,
                                      value="Automatic Detection", label="Language")
                dd_subformat = gr.Dropdown(["SRT", "WebVTT"], value="SRT", label="Subtitle Format")
            with gr.Row():
                cb_translate = gr.Checkbox(value=False, label="Translate to English?", interactive=True)
            with gr.Row():
                btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
            with gr.Row():
                tb_indicator = gr.Textbox(label="Output")
                btn_openfolder = gr.Button('?��').style(full_width=False)

            btn_run.click(fn=whisper_inf.transcribe_mic,
                          inputs=[mic_input, dd_model, dd_lang, dd_subformat, cb_translate], outputs=[tb_indicator])
            btn_openfolder.click(fn=lambda: open_folder("outputs"), inputs=None, outputs=None)
            dd_model.change(fn=on_change_models, inputs=[dd_model], outputs=[cb_translate])

        with gr.TabItem("T2T Translation"):  # tab 4
            with gr.Row():
                file_subs = gr.Files(type="file", label="Upload Subtitle Files to translate here",
                                     file_types=['.vtt', '.srt'])

            with gr.TabItem("NLLB"):  # sub tab1
                with gr.Row():
                    dd_nllb_model = gr.Dropdown(label="Model", value=nllb_inf.default_model_size,
                                                choices=nllb_inf.available_models)
                    dd_nllb_sourcelang = gr.Dropdown(label="Source Language", choices=nllb_inf.available_source_langs)
                    dd_nllb_targetlang = gr.Dropdown(label="Target Language", choices=nllb_inf.available_target_langs)
                with gr.Row():
                    btn_run = gr.Button("TRANSLATE SUBTITLE FILE", variant="primary")
                with gr.Row():
                    tb_indicator = gr.Textbox(label="Output")
                    btn_openfolder = gr.Button('?��').style(full_width=False)
                with gr.Column():
                    md_vram_table = gr.HTML(NLLB_VRAM_TABLE, elem_id="md_nllb_vram_table")

            btn_run.click(fn=nllb_inf.translate_file,
                          inputs=[file_subs, dd_nllb_model, dd_nllb_sourcelang, dd_nllb_targetlang],
                          outputs=[tb_indicator])
            btn_openfolder.click(fn=lambda: open_folder(os.path.join("outputs", "translations")), inputs=None, outputs=None)

currentParam = None

def fapi_progress(prog, desc):
    global currentParam
    print('\nfapi_progress:{} {}'.format(prog, desc))
    if currentParam is not None:
        do_callback(currentParam.input_file, 0, 'PROC{}'.format(prog), currentParam.uuid, currentParam.userid, currentParam.userno,
            currentParam.itemno, currentParam.orig_file, currentParam.callbackurl)
    else:
        print('fapi_progress cannot call do_callback, as currentParam is None')

def  mktempfn(uuidstr, file_name, input_data):
    decoded_data = base64.b64decode(input_data)
    file_name = file_name.replace('\\','/')
    name_parts = file_name.split('/')
    tempdir = './tempdir'
    try:
        os.mkdir(tempdir)
    except:
        pass
    tempdir += '/' + uuidstr # str(uuid.uuid4())
    try:
        os.mkdir(tempdir)
        tempfn = tempdir + '/' + name_parts[-1] # os.path.join(tempdir, name_parts[-1])
        with open(tempfn, 'wb') as f:
            f.write(decoded_data)
        return tempfn
    except:
        pass

    return ''

class inferParam() :
    def __init__(self, _pnt=0, _fname='', _dd_model='', _dd_lang='', _dd_subformat='SRT', _cd_translate=False, _uuid=''):
        self.point = _pnt
        self.fname = _fname
        self.dd_model = _dd_model
        self.dd_lang = _dd_lang
        self.dd_subformat = _dd_subformat
        self.cb_translate = _cd_translate
        self.uuid = _uuid

    '''
    def __init__(self):
        self.point = 0
        self.fname = ''
        self.dd_model = ''
        self.dd_lang = ''
        self.dd_subformat = ''
        self.cb_translate = ''
        self.uuid = ''
    '''

inputCount = 0
inputHead = 0
inputTail = 0
inputList = []

inputList = [inferParam()] * 100
lock = threading.Lock() # threading에서 Lock 함수 가져오기

def addInferInput(_pnt, _fname, _dd_model, _dd_lang, _dd_subformat, _cd_translate, _uuid, _userid, _userno, _itemno ):
    global inputCount, inputHead, inputTail, inputList
    ic = 0
    lock.acquire()
    ic = inputCount
    lock.release()
    if ic >= 100:
        return -1
    
    lock.acquire()
    inputList[inputHead] = inferParam(_pnt, _fname, _dd_model, _dd_lang, _dd_subformat, _cd_translate, _uuid,
        _userid, _userno, _itemno )
    inputHead += 1
    if inputHead >= 100:
        inputHead = 0
    inputCount += 1
    lock.release()

    return inputCount

def getInferInput():
    global inputCount, inputHead, inputTail, inputList
    ic = 0
    lock.acquire()
    ic = inputCount
    lock.release()
    if ic <= 0 :
        return None

    lock.acquire()
    ret = inputList[inputTail]
    inputTail += 1
    if inputTail >= 100:
        inputTail = 0
    inputCount -= 1
    lock.release()

    return ret

def getInferInputList():
    global inputCount, inputHead, inputTail, inputList
    ret = []
    lock.acquire()
    p = inputTail
    ic = inputCount
    while ic > 0 :
        ret.append(inputList[p])
        p += 1
        if p >= 100 :
            p = 0
        ic -= 1
    lock.release()

    return ret


def fthread_progress(prog, desc):
    fname = file_in_infer + '.{:03d}.prc'.format(int(prog))
    with open(fname, 'wt') as wf:
        wf.write('{:03d}'.format(prog))
    print('\nfapi_progress:{} {} {}'.format(file_in_infer, prog, desc))


def infer_thread():
    global file_in_infer

    while True:
        input = getInferInput()
        if input is None:
            time.sleep(0.01)
        else:
            file_in_infer = input.fname
            fileobj = {}
            fileobj['orig_name'] = input.fname
            fileobj['name'] = input.fname
            fileobjs = [fileobj]
            tb_indicator = whisper_inf.transcribe_file1(fileobjs, input.dd_model, input.dd_lang,
                input.dd_subformat, input.cb_translate, progress=fthread_progress)
            
            fname = file_in_infer + '.fin'
            with open(fname, 'wt') as wf:
                wf.write('fin')
            fname = file_in_infer + '.srt'
            with open(fname, 'wt', encoding='utf8') as wt:
                wt.write(tb_indicator)

@app.post('/transcribe_queue')  # http://192.168.7.188:5001/transcribe_file
async def fastapi_transcribe_queue(paramTranscribeFile: TranscribeFile):
    if paramTranscribeFile.input_data is None or paramTranscribeFile.input_data == '':
        print('fastapi_transcribe_queue No input')
        return {
            'result_type': -1,
            'result_msg': 'Fail: No input data',
        }
    
    tempfn = mktempfn(paramTranscribeFile.uuid, paramTranscribeFile.input_file, paramTranscribeFile.input_data)
    if tempfn == '':
        print('fastapi_transcribe_queue Temp save fail')
        return {  # 측정결과 코드 오류 메시지
            'result_type': -2,
            'result_msg': 'Fail: Temp save failed',
        }
    if paramTranscribeFile.callback is None or paramTranscribeFile.callback == '':
        print('fastapi_transcribe_queue callback is empty')
        return {  # 측정결과 코드 오류 메시지
            'result_type': -3,
            'result_msg': 'Fail: callback is empty',
        }

    #uuidstr = tempfn.split('/')[-2]
    print('uuidstr='+paramTranscribeFile.uuid)
    inputcnt = addInferInput(0, tempfn, paramTranscribeFile.dd_model,
        paramTranscribeFile.dd_lang, paramTranscribeFile.dd_subformat, paramTranscribeFile.cb_translate, paramTranscribeFile.uuid,
        paramTranscribeFile._userid, paramTranscribeFile._userno, paramTranscribeFile._itemno )
    if inputcnt > 0 :
        print('fastapi_transcribe_queue UUID={}'.format(paramTranscribeFile.uuid))
        return {  # 측정결과 코드 오류 메시지
            'result_type': 0,
            'result_msg': 'UUID:{}'.format(paramTranscribeFile.uuid)
        }

    print('fastapi_transcribe_queue adInferInput failes')
    return {
        'result_type': 0,
        'result_msg': 'Fail: add failed'
    }

@app.post('/transcribe_qget')
async def fastapi_transcribe_qget(paramTranscribeFile: TranscribeFile):
    if paramTranscribeFile.uuid is None or paramTranscribeFile.uuid == '':
        return {  # 측정결과 코드 오류 메시지
            'result_type': -1,
            'result_msg': 'No uuid',
        }
    
    tempdir = './tempdir/' + paramTranscribeFile.uuid
    templist = glob.glob(tempdir + '/*')
    fin = False
    proc = 0
    srtfn = ''
    for f in templist:
        if f.endswith('.fin'):
            print('{} found'.format(f))
            fin = True
            break
        if f.endswith('.srt'):
            print('{} found'.format(f))
            srtfn = f

        if f.endswith('.prc'):
            prc = int(f[-7:-4])
            if prc > proc:
                proc = prc
    if fin :
        with open(srtfn, 'rt', encoding='utf8') as inf:
            srt_text = inf.read()
        #for f in templist:
        #    os.remove(f)
        #os.rmdir(tempdir)

        return {  # 측정결과 코드 오류 메시지
            'result_type': 0,
            'result_msg': srt_text,
        }

    # not finished return proc
    return {
        'result_type': 0,
        'result_msg': 'PROC {}'.format(proc),
    }

@app.post('/transcribe_qlist')
async def fastapi_transcribe_qlist():
    qlist = getInferInputList()
    return {
        'result_type': 0,
        'result_msg': ','.join(qlist)
    }



@app.post('/transcribe_file')  # http://192.168.7.188:5001/transcribe_file
async def fastapi_transcribe_file(paramTranscribeFile: TranscribeFile):
    global currentParam

    fileobj = {}
    fileobj['orig_name'] = paramTranscribeFile.input_file
    fileobj['name'] = paramTranscribeFile.input_file
    fileobjs = [fileobj]
    currentParam = paramTranscribeFile
    if paramTranscribeFile.dd_model is None or paramTranscribeFile.dd_model == '':
        paramTranscribeFile.dd_model = 'large-v2'
    if paramTranscribeFile.dd_subformat is None or paramTranscribeFile.dd_subformat == '':
        paramTranscribeFile.dd_subformat = 'SRT'
    if paramTranscribeFile.cb_translate == '':
        paramTranscribeFile.cb_translate = False
    tb_indicator = whisper_inf.transcribe_file1(fileobjs, paramTranscribeFile.dd_model,
        paramTranscribeFile.dd_lang, paramTranscribeFile.dd_subformat, paramTranscribeFile.cb_translate, progress=fapi_progress)
    
    return {  # 측정결과 코드 오류 메시지
        'result_type': 0,
        'result_msg': tb_indicator,
    }

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

server_thread = threading.Thread(target=run_server)
server_thread.start()

if args.share:
    block.launch(share=True)
else:
    block.launch()

