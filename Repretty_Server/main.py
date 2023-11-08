from flask import Flask, request, abort  # Flask framework
from flask_restx import Api, Resource #Flask rest api

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import openai
import requests
from functools import cache
from dotenv import load_dotenv
import os


app = Flask(__name__)
api = Api(app)

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

@cache #Cached Model
def get_model():
    return SentenceTransformer('jhgan/ko-sroberta-multitask')

"""
    dataset 로드가 상당히 오래 걸리는데 미리 json형태로 가공해서 리로드 할 수 있게 하면 괜찮을것 같네용
    이렇게 해두면 /reload endpoint나 관리자 페이지에서 서버에 파일 로드해놓고 불러오면 되서
"""
def load_dataset():
    global train_data
    print("Loading datasets..")
    train_data = pd.read_csv('https://docs.google.com/spreadsheets/d/1qucS86xmwJ-yiRBVI44m72sPlsz73-LMQrQPya46JQA/export?format=csv')
    train_data.head(20)
    # NaN 제거하기
    train_data.dropna(axis=1)
    print("Applying Embedding..")
    train_data['embedding'] = train_data.apply(lambda row: get_model().encode(row.Q), axis=1)
    print("dataset load end.")


def find_dataset_answer(question):
    global train_data
    model = get_model()
    embedding = model.encode(question)
    train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    answer = train_data.loc[train_data['score'].idxmax()]['A']
    score = train_data.loc[train_data['score'].idxmax()]['score']
    return answer, score

def request_chatgpt(question):
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role" : "user", "content": question}]
    )
    reply = chat.choices[0].message.content
    return reply
#/Question endpoint. post요청으로 질문 요청시 response
@api.route('/question')
class Question(Resource):
    """
    Request
    {question : "blah blah"}
    Response (text/event-stream)
    {data : "안"} \n\n   {data : "녕"} \n\n ...

    event-stream 미사용시
    { recommend : true, answer : "안녕" } 이렇게 바로 반환해서 주제추천 할지 말지 결정할 필요 없긴한데..
    chat-gpt 생성 결과값을 stream으로 이욯하기 위해선 event-stream 반환 필수.. stream에서 json 반환시 깔끔하지 않음..ㅠㅠ
    """
    def post(self):
        body = request.get_json(force=True)
        print(body)
        question = body['question']  # 질문
        if question is None:
            abort(400, "need question parameter")
        else:
            answer, score = find_dataset_answer(question)
            print("score : " + str(score))
            if score <= 0.64:
                answer = request_chatgpt(question)
            return {"answer": answer}

@api.route("/recommend")
class Recommend(Resource):
    def post(self):
        question = request.json.get['question'] #질문
        if question is None:
            abort(400, "need question parameter")





if __name__ == "__main__":
    load_dotenv() #env파일로부터 api키 흭득
    client = OpenAI(api_key=os.environ.get("openai.key")) #왜 바뀐거지..
    load_dataset()  # init시 데이터셋 로드
    app.run(debug=False, host='127.0.0.1', port=80)