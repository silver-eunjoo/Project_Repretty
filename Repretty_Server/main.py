import flask
from flask import Flask, request, abort, Response  # Flask framework
from flask_restx import Api, Resource  # Flask rest api

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm

from openai import OpenAI, NoneType
from sentence_transformers import SentenceTransformer
import openai
import requests
from functools import cache
from dotenv import load_dotenv
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Repretty API Server')

message_record = []


def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


@cache  # Cached Model
def get_model():
    return SentenceTransformer('jhgan/ko-sroberta-multitask')


"""
    dataset 로드가 상당히 오래 걸리는데 미리 json형태로 가공해서 리로드 할 수 있게 하면 괜찮을것 같네용
    이렇게 해두면 /reload endpoint나 관리자 페이지에서 서버에 파일 로드해놓고 불러오면 되서
"""


def load_dataset():
    file_path = str(os.getcwd()) + "/dataset.pkl"
    print("Loading datasets..")
    if os.path.isfile(file_path):
        local_data = pd.read_pickle(file_path)  # embeding까지 적용된 pickle로드 pickle사용시 파이썬 객체를 그대로 저장해서 불러오기 좋음
    else:
        local_data = pd.read_csv(
            'https://drive.google.com/u/1/uc?id=1YdeNo_nBzSF-kwdvVHSQrsEF9QpyMaBE&export=download')  # 다운로드 링크로 넣어야 csv파일이 정상적으로 로드 되네용
        local_data.head(20)
        # NaN 제거하기
        local_data.dropna(axis=1)
        print("Applying Embedding..")
        local_data['embedding'] = local_data.apply(lambda row: get_model().encode(row.Q), axis=1)
        print("dataset load end.")
        local_data.to_pickle(file_path)
        print("Saving file completed.")

    return local_data


def find_dataset_answer(question):
    model = get_model()
    embedding = model.encode(question)
    train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    answer = train_data.loc[train_data['score'].idxmax()]['A']
    score = train_data.loc[train_data['score'].idxmax()]['score']
    return answer, score


def request_chatgpt(question: list, system: list = None, assistant: list = None, stream=False, maintain_record=False):
    # 파라미터 입력
    # TODO message 이력 유지?
    message = []
    if system is not None:
        for msg in system:
            message.append({"role": "system", "content": msg})
    if assistant is not None:
        for msg in assistant:
            message.append({"role": "assistant", "content": msg})
    for msg in question:
        message.append({"role": "user", "content": msg})
    # api 호출
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=message, stream=stream
    )
    if stream:
        return chat  # stream이 활성화일경우 제너레이터 반환
    else:
        reply = chat.choices[0].message.content  # 일반 답변 반환
        return reply


# /Question endpoint. post요청으로 질문 요청시 response
@api.route('/question')
class Question(Resource):
    """
    Request
    {"question" : "blah blah"}
    Response (text/event-stream)
    {"data" : "안"} \n\n   {"data" : "녕"} \n\n ...

    event-stream 미사용시
    { "recommend" : true, "answer" : "안녕" } 이렇게 바로 반환해서 주제추천 할지 말지 결정할 필요 없긴한데..
    chat-gpt 생성 결과값을 stream으로 이욯하기 위해선 event-stream 반환 필수.. stream에서 json 반환시 깔끔하지 않음..ㅠㅠ

    11/15 : json의 형식으로 event-stream에 전송할경우 상당히 불안정해짐.
    """

    def post(self):
        """요청된 Question paramter를 이용하여 답변을 text/event-stream으로 반환합니다."""
        body = request.get_json(force=True)
        print(body)
        if "question" not in body:
            abort(400, "need question parameter")
        question = body['question']  # 질문
        answer, score = find_dataset_answer(question)
        print("score : " + str(score))
        if score <= 0.64:
            def stream_gpt():
                for message in request_chatgpt(question=list([question]), stream=True):
                    text = message.choices[0].delta.content
                    if type(text) != NoneType and len(text):
                        print(text)
                        yield text

            return Response(flask.stream_with_context(stream_gpt()), mimetype='text/event-stream')
        else:
            return Response(flask.stream_with_context((char for char in answer)),
                            mimetype='text/event-stream')  # 한글자식 리턴


@api.route("/recommend")
class Recommend(Resource):
    def post(self):
        """요청된 question과 answer를 이용하여 예상되는 다음 질문을 반환합니다. 갱년기와 관련이 없는 주제일경우 빈 리스트가 반환됩니다."""
        body = request.get_json(force=True)
        print(body)
        if "question" not in body or "answer" not in body:
            abort(400, "need question and answer parameter")
        question = body['question']  # 질문
        answer = body['answer']  # 답변
        _, score = find_dataset_answer(question)
        # 0.64 이하의 값은 추천하지 않음. 그보다 높은 값일경우 chatgpt에 요청함
        if score <= 0.64:
            return {"isRecommend": False, "list": []}
        else:
            # 영어로 요청시 응답속도 빨라진다는 이야기가.. -> 더 길어져서 느려짐
            return {"isRecommend": True, "list": request_chatgpt(
                question=list(
                    ["질문이 '" + question + "'이고 답변이 '" + answer + "' 일때 다음 질문으로 할만한 문장을 간단하게 3개 제시해줘."])).split(
                "\n")}


if __name__ == "__main__":
    load_dotenv()  # env파일로부터 api키 흭득
    client = OpenAI(api_key=os.environ.get("openai.key"))  # 왜 바뀐거지..
    train_data = load_dataset()  # init시 데이터셋 로드
    app.run(debug=False, host='127.0.0.1', port=80)
