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

def do_callback(input_fname,result_type,result_msg,input_uuid,input_userid,input_userno,input_itemno,input_orig_file,input_callbackurl):
    print('do_callback is called.')
    try:
        callback_data = {
            "input_file": input_fname,
            "result_type": result_type,
            "result_msg": result_msg,
            "uuid": input_uuid,
            "userid" : input_userid,
            "userno" : input_userno,
            "itemno" : input_itemno,
            "orig_file" : input_orig_file
        }
        callbackurl = input_callbackurl
        print('callbackurl={}'.format(callbackurl))
        callback_response = requests.post(callbackurl, json=callback_data)
        print(callback_response)
    except Exception as ex:
        print('do_infer:exception:'+str(ex))
