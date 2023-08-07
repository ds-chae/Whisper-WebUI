#-*- coding: utf-8 -*-
import os
from ui.htmls import *
import argparse
import requests

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
    userno : Union[int, None] = None
    itemno : Union[int, None] = None
    orig_file : Union[str, None] = None

from datetime import datetime

import uuid

app = FastAPI()


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


def fapi_progress(prog, desc):
    print('\nfapi_progress:{} {}'.format(prog, desc))

def  mktempfn(file_name, input_data):
    decoded_data = base64.b64decode(input_data)
    file_name = file_name.replace('\\','/')
    name_parts = file_name.split('/')
    tempdir = './tempdir'
    try:
        os.mkdir(tempdir)
    except:
        pass
    tempdir += '/' + str(uuid.uuid4())
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
    def __init__(self, _pnt=0, _fname='', _dd_model='', _dd_lang='', _dd_subformat='SRT', _cd_translate=False,
                 _uuid='', _callbackurl='', _userid='', _userno=-1, _itemno=-1, _orig_file='' ):
        self.point = _pnt
        self.fname = _fname
        self.dd_model = _dd_model
        self.dd_lang = _dd_lang
        self.dd_subformat = _dd_subformat
        self.cb_translate = _cd_translate
        self.uuid = _uuid
        self.callbackurl = _callbackurl
        self.userid = _userid
        self.userno = _userno
        self.itemno = _itemno
        self.orig_file = _orig_file
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

def addInferInput(_pnt, _fname, _dd_model, _dd_lang, _dd_subformat, _cd_translate, _uuid, _callbackurl,
        _userid, _userno, _itemno, _orig_file ):
    global inputCount, inputHead, inputTail, inputList
    ic = 0
    lock.acquire()
    ic = inputCount
    lock.release()
    if ic >= 100:
        return -1
    
    lock.acquire()
    inputList[inputHead] = inferParam(_pnt, _fname, _dd_model, _dd_lang, _dd_subformat, _cd_translate,
        _uuid, _callbackurl, _userid, _userno, _itemno, _orig_file )
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


file_in_infer = ''

def fthread_progress(prog, desc):
    fname = file_in_infer + '.{:03d}.prc'.format(int(prog))
    with open(fname, 'wt') as wf:
        wf.write('{:03d}'.format(prog))
    print('\nfapi_progress:{} {} {}'.format(file_in_infer, prog, desc))

def do_infer(input):
    file_in_infer = input.fname
    print('appgate: request to transcribe {}'.format(input.fname))
    # Define new data to create
    new_data = {
        "input_file": input.fname,
        "input_data": "",
        "dd_model": input.dd_model,
        "dd_lang": input.dd_lang,
        "dd_subformat": input.dd_subformat,
        "cb_translate": input.cb_translate,
        "callbackurl" : input.callbackurl,
        "uuid": input.uuid,
        "userid": input.userid,
        "userno": input.userno,
        "itemno": input.itemno,
        "orig_file": input.orig_file
    }

    trans_done = True
    # The API endpoint to communicate with
    url_post = "http://localhost:8000/transcribe_file"
    # A POST request to the API
    try:
        post_response = requests.post(url_post, json=new_data)
        # Print the response
        post_response_json = post_response.json()
        #print(post_response_json)
        result_type = post_response_json['result_type']
        print('type={}'.format(result_type))
        result_msg = post_response_json['result_msg']
        #print('msg={}'.format(result_msg))
        fname = file_in_infer + '.fin'
        with open(fname, 'wt') as wf:
            wf.write('fin')
        fname = file_in_infer + '.srt'
        with open(fname, 'wt', encoding='utf8') as wt:
            wt.write(result_msg)
    except Exception as ex:
        print('do_infer:exception:' + str(ex))
        trans_done = False

    if trans_done:
        do_callback(input.fname,result_type,result_msg,input.uuid,input.userid,input.userno,
                        input.itemno,input.orig_file, input.callbackurl)


def infer_thread():
    global file_in_infer

    while True:
        input = getInferInput()
        if input is None:
            time.sleep(0.01)
            continue

        do_infer(input)


@app.post('/transcribe_queue')  # http://192.168.7.188:5001/transcribe_file
async def fastapi_transcribe_queue(paramTranscribeFile: TranscribeFile):
    if paramTranscribeFile.input_data is None or paramTranscribeFile.input_data == '':
        print('appgate:fastapi_transcribe_queue No input')
        return {
            'result_type': -1,
            'result_msg': 'Fail: No input data',
        }
    
    tempfn = mktempfn(paramTranscribeFile.input_file, paramTranscribeFile.input_data)
    if tempfn == '':
        print('fastapi_transcribe_queue Temp save fail')
        return {  # 측정결과 코드 오류 메시지
            'result_type': -2,
            'result_msg': 'Fail: Temp save failed',
        }
    
    uuidstr = tempfn.split('/')[-2]
    print('uuidstr='+uuidstr)
    inputcnt = addInferInput(0, tempfn, paramTranscribeFile.dd_model,
            paramTranscribeFile.dd_lang, paramTranscribeFile.dd_subformat, paramTranscribeFile.cb_translate, uuidstr,
            paramTranscribeFile.callbackurl,
            paramTranscribeFile.userid, paramTranscribeFile.userno, paramTranscribeFile.itemno,
            paramTranscribeFile.input_file)
    if inputcnt > 0 :
        print('fastapi_transcribe_queue UUID={}'.format(uuidstr))
        return {  # 측정결과 코드 오류 메시지
            'result_type': 0,
            'result_msg': 'UUID:{}'.format(uuidstr)
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




if __name__ == "__main__":
    t1 = threading.Thread(target=infer_thread)
    t1.start()

    uvicorn.run(app, host="0.0.0.0", port=8001)
