#-*- coding: utf-8 -*-
import os
from ui.htmls import *
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


from datetime import datetime

import uuid

# main.py

from sentence_transformers import SentenceTransformer, util

bi_encoder = SentenceTransformer('nq-distilbert-base-v1')

### Create corpus embeddings containing the wikipedia passages
### To keep things summaraized, we are not going to show the code for this part

def predict(question):
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    # transform hits into a list of dictionaries, and obtain passages with corpus_id
    results = [
        {
            "score": hit["score"],
            "title": passages[hit["corpus_id"]][0],
            "text": passages[hit["corpus_id"]][1],
        }
        for hit in hits
    ]

    return results

# main.py

import gradio as gr

def gradio_predict(question: str):
    results = predict(question) # results is a list of dictionaries

    best_result = results[0]

    # return a tuple of the title and text as a string, and the score as a number
    return f"{best_result['title']}\n\n{best_result['text']}", best_result["score"]

# main.py

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        label="Ask a question", placeholder="What is the capital of France?"
    ),
    outputs=[gr.Textbox(label="Answer"), gr.Number(label="Score")],
    allow_flagging="never",
)

demo.launch()

# main.py

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Request(BaseModel):
    question: str


class Result(BaseModel):
    score: float
    title: str
    text: str


class Response(BaseModel):
    results: typing.List[Result] # list of Result objects


@app.post("/predict", response_model=Response)
async def predict_api(request: Request):
    results = predict(request.question)
    return Response(
        results=[
            Result(score=r["score"], title=r["title"], text=r["text"])
            for r in results
        ]
    )

# main.py

# mounting at the root path
app = gr.mount_gradio_app(app, demo, path="/")
