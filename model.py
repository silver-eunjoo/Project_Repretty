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

openai.api_key = "KEY" #ChatGPT KEY 받아오기
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

#Gdrive Path
train_data = pd.read_csv('/content/gdrive/MyDrive/project_0402/Data/DB.csv') #이건 Google Spreadsheet로 공유해서 다시 경로 설정하기
train_data.head(20)

#NaN 제거하기
train_data.dropna(axis=1)

# Add 'embeddingn' row, 질문의 임베딩값 저장하기
train_data['embedding'] = train_data.apply(lambda row: model.encode(row.Q), axis = 1)

# calculate cosine similarity
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

train_data

# 나중에 CHATGPT 연동해서 더 심화할 때 필요할 듯..?
# def calculate_distance(text, model):
#     embedding = model.encode(text) #입력받은 text의 임베딩값 구하기
#     train_distance['distance'] = train_distance['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze()) #저장된 임베딩값과 입력받은 임베딩 값의 거리 구해서 distance column에 저장
#     return train_distance

# # 거리가 가장 먼 질문
# def get_max_distance_answer(df):
#     answer = qa_dict[df.loc[df['distance'].idxmax(), 'Q']]
#     return answer

#df = df[~df['챗봇'].isna()] # isna : NaN 제거


// 현재 챗봇

def return_answer(question) :
    embedding = model.encode(question)
    train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return train_data.loc[train_data['score'].idxmax()]['A']


return_answer("갱년기에 좋은 약이 있을까?")
#갱년기 영양제를 복용하면 여성의 갱년기 증상을 완화하고 여성의 건강을 유지하는 데 도움을 줍니다. 갱년기 영양제는 여성의 건강을 위해 다양한 성분이 함유되어 있습니다. 이 성분들은 갱년기 증상을 완화하고 여성의 건강을 유지하는 데 도움을 줍니다. 갱년기 영양제를 복용하기 전에는 꼭 의사와 상담하시는 것이 좋습니다. 의사의 처방에 따라 적절한 갱년기 영양제를 복용하시는 것이 좋습니다. 갱년기 영양제 추천으로는 칼슘 영양제, 식이섬유, 비타민D 영양제, 홍삼, 백수오, 석류 등이 있습니다 .

return_answer("요즘 잠을 못 자겠어")
#갱년기 불면증은 갱년기 여성들이 겪는 수면장애로, 수면의 질과 양이 감소하는 것이 특징입니다.\x80갱년기 불면증의 주요 증상은 밤에 자주 깨어나고 잠들기가 어렵고, 잠이 너무 얕아서 일어나면 피로감이 남아있을 수 있으며, 잠을 자면서도 불쾌감과 긴장감이 남아 불안감을 느낄 수도 있습니다.

return_answer("기분이 좋았다가 안 좋았다가 감정이 요동쳐")
#기분이 오락가락하는 것은 갱년기의 일부 증상 중 하나일 수 있습니다. 갱년기는 여성의 난소 기능이 감소하면서 일어나는 생리기능의 변화로 인해 호르몬 수준이 변동적으로 변하게 되어 여러 가지 증상이 나타날 수 있습니다. 이 중에서도 기분이 변화하는 증상은 갱년기의 일부 증상 중 하나로 알려져 있습니다. 이러한 증상은 감정적인 변화, 불안감, 우울증 등으로 나타날 수 있습니다.\n그러나 이러한 증상이 갱년기 때문인지 정확하게 판단하기 위해서는 전문가의 진단이 필요합니다. 따라서, 증상이 계속되거나 심해진다면 전문가와 상담하시는 것이 좋습니다.

return_answer("갱년기를 극복할 수 있는 방법이 있을까?")
#갱년기 증상을 완화하기 위한 방법은 다양합니다. 아래는 일반적으로 권장되는 방법들입니다.\n\n호르몬 치료: 에스트로겐 또는 프로게스테론을 보충하여 갱년기 증상을 완화하는 방법입니다. 하지만 호르몬 치료는 부작용이 있을 수 있으므로, 전문의와 상의하고 치료를 받는 것이 좋습니다.\n\n생활습관 개선: 건강한 생활습관을 유지하는 것이 중요합니다. 충분한 수면과 휴식을 취하고, 금연 및 음주를 자제하며, 스트레스를 관리하는 것이 도움이 됩니다.\n\n식이요법: 갱년기 때는 식이요법도 중요합니다. 칼슘과 비타민 D가 풍부한 식품을 먹는 것이 좋습니다. 또한 식이섬유가 풍부한 채소와 과일, 단백질을 충분히 섭취하는 것도 중요합니다.\n\n운동: 운동은 갱년기 증상을 완화하는데 도움이 됩니다. 꾸준한 유산소 운동과 근력운동을 하는 것이 좋습니다.\n\n치료적 요법: 갱년기 증상 중 호흡곤란, 땀이 많이 나는 것, 가슴 통증 등 일부 증상은 치료적 요법으로 완화될 수 있습니다. 따라서 전문의와 상의하여 적절한 치료를 받는 것이 좋습니다.\n\n보조 요법: 갱년기 증상을 완화하기 위해 천연 보조 요법도 사용될 수 있습니다. 예를 들어, 대추 추출물, 대황 추출물 등의 보조 요법이 있습니다. 그러나 보조 요법은 전문의와 상의하고 사용하는 것이 좋습니다.\n\n이러한 방법들은 각각의 개인에 따라 효과가 다를 수 있으므로, 각 개인의 상황에 맞게 선택하고 조절하는 것이 중요합니다. 또한 갱년기 증상이 심각한 경우, 전문의와 상의하고 치료를 받는 것이 좋습니다.

return_answer("으라차차")
#케겔운동은 골반저근의 운동 방법으로 항문에 5~10초 정도 힘을 준 뒤에 서서히 힘을 빼는 동작을 반복하는 것입니다. 이 동작을 반복할 경우 요도에서 항문 부위에 수축을 담당하는 치골미골근이 강화됩니다.\n케겔운동을 하기 전에는 복부와 대퇴근을 펴고, 무릎을 굽히지 않고 서서 쉬운 자세를 취합니다. 그리고 항문을 수축하면서 5초간 유지한 뒤, 5초간 풀어주는 것을 10회 반복합니다.\n케겔운동을 하는 것이 처음이라면, 일어나서 하는 것보다 누워서 하는 것이 더 쉬울 수 있습니다. 또한, 케겔운동을 하는 동안 복부나 대퇴근을 이용하여 동작을 보조할 수 있습니다.
#이런 관계없는 질문은 ChatGPT 연동해서 답변 받아보도록 하기


#아래는 아직 실험해보지 않았음. 
#추후에 chat gpt 연동하고, cosine similarity 기반으로 일정 값 이상일 때는 DB를 기초로 한 답변 제공하고, 일정 값 아래일 때는 ChatGPT 답변으로 받기
#그 전까지는 아래는 주석처리 후 return_answer()함수로 실험해보도록 함. 

while True:
  text = input('유저: ')
  embedding = model.encode(text)
  train_data['score'] = train_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
  print(train_data['score'])
  answer = train_data.loc[train_data['score'].idxmax()]['A']
  print("highest :")
  print(train_data.loc[train_data['score'].idxmax()]['score'])

  if train_data.loc[train_data['score'].idxmax()]['score']< 0.6 :
      chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", answer=answer
        )
      reply = chat.choices[0].answer.content
      print(reply)
      #response = openai.Completion.create(engine="davinci", prompt=text, max_tokens=60, n=1, stop=None, temperature=0.5)
      #answer = response.choices[0].text.strip()
  else :
    print(answer)

