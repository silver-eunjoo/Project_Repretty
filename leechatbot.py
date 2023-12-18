# -*- coding: utf-8 -*-
"""LeeChatbot.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vRpd_qTXivje3afV6sppIX8ICKew8rcB
"""

!pip install sentence_transformers
!pip install openai

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import requests

from google.colab import drive
drive.mount("/content/gdrive")

#api 키 에러나서 재발급 받았습니당
openai.api_key = "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" #ChatGPT KEY 받아오기
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)

#Gdrive Path
train_data = pd.read_csv('/content/gdrive/MyDrive/project_0402/Data/DB.csv')
#pandas spreadsheet
# train_data = pd.read_csv('https://docs.google.com/spreadsheets/d/1qucS86xmwJ-yiRBVI44m72sPlsz73-LMQrQPya46JQA/export?format=csv')
train_data.head(20)

#NaN 제거하기
train_data.dropna(axis=1)

# Add 'embeddingn' row, 질문의 임베딩값 저장하기
train_data['embedding'] = train_data.apply(lambda row: model.encode(row.Q), axis = 1)

# calculate cosine similarity
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

train_data

# def calculate_distance(text, model):
#     embedding = model.encode(text) #입력받은 text의 임베딩값 구하기
#     train_distance['distance'] = train_distance['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze()) #저장된 임베딩값과 입력받은 임베딩 값의 거리 구해서 distance column에 저장
#     return train_distance

# # 거리가 가장 먼 질문
# def get_max_distance_answer(df):
#     answer = qa_dict[df.loc[df['distance'].idxmax(), 'Q']]
#     return answer

#df = df[~df['챗봇'].isna()] # isna : NaN 제거

def return_answer(question) :
    embedding = model.encode(question)
    train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return train_data.loc[train_data['score'].idxmax()]['A']

return_answer("갱년기에 좋은 약이 있을까?")

return_answer("요즘 잠을 못 자겠어")

return_answer("기분이 좋았다가 안 좋았다가 감정이 요동쳐")

return_answer("갱년기를 극복할 수 있는 방법이 있을까?")

return_answer("으라차차")

pip install openai==0.28

texts = [{"role" : "user", "content": ""},]

while True:
  text = input('유저: ')
  embedding = model.encode(text)
  train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
  print(train_data['score'].idxmax()) # score가 가장 높은 행 번호 출력
  answer = train_data.loc[train_data['score'].idxmax()]['A'] # score 가장 높은 행의 Answer 리턴
  print("highest :")
  print(train_data.loc[train_data['score'].idxmax()]['score']) # 가장 큰 cos_sim 수치를 나타냄. = high_dist

  # print(f"general chatbot : {answer}")
  # print(text)
  high_dist = train_data.loc[train_data['score'].idxmax()]['score']

  texts.append({"role":"user", "content": text},)

  print(texts)

  if high_dist <= 0.64 :
      chat = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=texts
        )
      reply = chat.choices[0].message.content
      print(f"Repretty chatbot : {reply}")
      texts.append({"role": "assistant", "content" : reply})

  else :
    print(f"Repretty chatbot : {answer}")

# 유저: 남편이랑 사이가 너무 안 좋아
# 236
# highest :
# 0.6225272
# Repretty chatbot : 갱년기 불면증에 효과적인 방법은 크게 3가지로 나눌 수 있습니다. 첫째, 생활 습관 개선으로 규칙적인 생활 패턴, 건강한 식습관, 운동, 스트레스 관리 등을 실천하는 것이 중요합니다. 둘째, 수면 환경 개선으로 적절한 온도와 조용한 환경을 조성하고, 침실에서 스마트폰이나 TV 시청을 자제하는 것이 좋습니다. 셋째, 전문가 상담과 약물 치료가 있습니다. 갱년기 불면증은 정신적, 신체적인 문제가 함께 작용하여 발생하기 때문에, 전문가 상담을 받으면 치료 및 조언을 받을 수 있습니다. 또한, 약물 치료를 통해 증상을 완화하는 것도 방법 중 하나입니다. 이러한 방법들을 함께 시도해보면, 갱년기 불면증을 완화할 수 있으며, 더 나은 수면을 취할 수 있습니다.


# 유저: 휴식을 취하는 방법
# 426
# highest :
# 0.68028075
# Repretty chatbot : 몸과 마음을 편안하게 하는 방법은 다양합니다. 스트레칭을 하거나 운동을 하면 몸의 근육을 늘리고 이완시켜서 혈액순환을 원활하게 하여 피로감을 줄이는 데 매우 효과적입니다1. 호흡에 집중하거나 명상을 하면 마음의 안정을 취할 수 있습니다2. 일기를 쓰거나 그림을 그리는 것도 마음의 안정에 도움이 됩니다2. 또한, 아로마 테라피를 이용하거나 차를 마시는 것도 좋은 방법입니다3. 이 외에도 다양한 방법이 있으니 자신에게 맞는 방법을 찾아보시는 것도 좋을 것 같습니다.



# 현재 high_dist 기준점을 0.6으로 두고 있음.

# 1번, 2번 모두 0.6 이상이라 DB based 답변임.
# 근데 1번 질문에 대한 답변은 어색한 것을 볼 수 있음. (high_dist : 0.62정도)

# 2번 질문에 대한 답은 자연스러운 것 같음. (high_dist : 0.68)

# => 기준점을 0.64로 조정함.

#ChatGPT 주제 유사한지?
def isSimilar(sentence, subject) :
  chat = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "system", "content": 'assistant는 갱년기에 관한 상담을 해주는 상담가이다'}, {"role" : "user", "content" : "'{0}'라는 질문과 {1}가 어느정도 관련성이 있을경우 'true'라고만 답해주고 그렇지 않거나 판단할 수 없다면 'false'라고만 답해줘".format(sentence, subject)}]
        )
  reply = chat.choices[0].message.content
  print(f"Repretty chatbot : {reply}")

isSimilar("너무 우울하네요", "갱년기")

isSimilar("나 혼자 있고 싶어", "갱년기")

isSimilar("으라차차차차", "갱년기")

isSimilar("행복한 하루 되세요", "갱년기")

isSimilar("지금 몇시야?", "갱년기")

# 주제가 갱년기와 관련이 있는지 판단하는 함수

def isSimilar(question):
  theme = [{"role" : "user", "content": ""},]
  question = "\"" + question + "\"라는 말은 갱년기와 관련이 있을까? 조금이라도 관련있으면 True, 그게 아니라면 False로만 대답해줘." #질문을 조금만 바꿔도 값이 달라질 수 있으니 유의
  # print(question)
  theme.append({"role":"user", "content": question},)

  # print(theme)
  chat = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=theme
        )
  reply = chat.choices[0].message.content
  # print(f"Repretty chatbot : {reply}")

  if(reply == "True"):
    return 1
  else :
    return 0

num = isSimilar("지금 몇시야?")
num

isSimilar("너무 우울하네요")

isSimilar("나 혼자 있고 싶어")

texts = [{"role" : "user", "content": ""},]

while True:
  text = input('유저: ')
  embedding = model.encode(text)
  train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
  print(train_data['score'].idxmax()) # score가 가장 높은 행 번호 출력
  answer = train_data.loc[train_data['score'].idxmax()]['A'] # score 가장 높은 행의 Answer 리턴
  print("highest :")
  print(train_data.loc[train_data['score'].idxmax()]['score']) # 가장 큰 cos_sim 수치를 나타냄. = high_dist

  print(f"general chatbot : {answer}")
  high_dist = train_data.loc[train_data['score'].idxmax()]['score']

  texts.append({"role":"user", "content": text},)

  print(texts)
  questions = [{"role" : "user", "content": ""},]
  question = "\"" + text + "\"에 후행될만한 질문을 갱년기와 관련해서 3개 예측해줘"
  questions.append({"role":"user", "content": question},)

  #similar 변수에 저장
  similar = isSimilar(text)
  if similar ==1 :
      chat = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=questions
        )
      reply = chat.choices[0].message.content
      print(f"예측 질문 : {reply}")

  if high_dist <= 0.64 :
      chat = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=texts
        )
      reply = chat.choices[0].message.content
      print(f"Repretty chatbot : {reply}")
      texts.append({"role": "assistant", "content" : reply})

  else :
    print(f"Repretty chatbot : {answer}")

  if high_dist > 0.64 or similar :
    print("또 어떤것이 궁금하신가요?")
  else :
    print("갱년기를 해결할 수 있는 방법에 대해서는 궁금하지 않으신가요?")


# 관련있을 때 질문 3가지 받는 것까지 구현 완료
# 질문 refining, tokenizing 안 했음.
# 1. ~ 2. ~ 3. ~ 여기서 질문만 따로 이차원 배열에 저장하는 코드 짜야함.
# 질문 고르거나 입력하는 코드 짜야함.
# 아악! 너무 많아.
