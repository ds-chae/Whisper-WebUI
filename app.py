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

app = FastAPI()

class TranscribeFile(BaseModel):
    input_file : Union[str, None] = None
    dd_model : Union[str, None] = None
    dd_lang : Union[str, None] = None
    dd_subformat : Union[str, None] = None
    cb_translate : Union[str, None] = None

class PredictionRequest(BaseModel):
    input: str

class PredictionResponse(BaseModel):
    output: str

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Add your prediction logic here
    output = "Your prediction result"

    return PredictionResponse(output=output)

from datetime import datetime

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

def fapi_progress(prog, desc):
    print('\nfapi_progress:{} {}'.format(prog, desc))

@app.post('/transcribe_file')  # http://192.168.7.188:5001/transcribe_file
async def fastapi_transcribe_file(paramTranscribeFile: TranscribeFile):
    fileobj = {}
    fileobj['name'] = paramTranscribeFile.input_file
    fileobj['orig_name'] = paramTranscribeFile.input_file
    fileobjs = [fileobj]
    tb_indicator = whisper_inf.transcribe_file1(fileobjs, paramTranscribeFile.dd_model,
        paramTranscribeFile.dd_lang, paramTranscribeFile.dd_subformat, paramTranscribeFile.cb_translate, progress=fapi_progress)
    return {  # 측정결과 코드 �?메시지
        'result_type': 0,
        'result_msg': tb_indicator,
    }

import threading
from fastapi import FastAPI, Request
import uvicorn

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

server_thread = threading.Thread(target=run_server)
server_thread.start()

if args.share:
    block.launch(share=True)
else:
    block.launch()

