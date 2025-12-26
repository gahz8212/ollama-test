from __future__ import annotations

import uuid
from typing import List, Optional, Dict, Any

import requests
import chromadb
import fitz  # PyMuPDF
from rdflib.plugins.stores.berkeleydb import results_from_key_func


class ChromaRAG:
###############
# 1. 설정부분.
    # ollama설정
    # 올라마 url : http://localhost:11434
    # 인베딩 모델 : nomic-embed-text
    # 생성형 모델 : llama3:1b, gemma3:1bf

    def __init__(self,
                 chroma_dir : str = './chroma_data',
                 collection_name : str = 'rag_docs',
                 ollama_base_url : str = 'http://localhost:11434',
                 embed_model : str = 'nomic-embed-text',
                 gen_model :str = 'gemma3:1b'):

         # Ollama 설정
        self.ollama_base_url = ollama_base_url
        self.embed_model = embed_model
        self.gen_model = gen_model

        # chroma설정
        # 폴더만든 것에 chroma db연결함.
        # --> chroma_data
        # collection(table, 폴더)를 생성함.
        # --> rag_docs
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name) #rag_docs
        self.collection2 = self.client.get_or_create_collection(name=collection_name + str(2)) #rag_docs2
    def __str__(self):
        return str(self.client) + " " + self.embed_model + " " + self.gen_model + " " + str(self.collection)+ " " + str(self.collection2)

###############
# 2. 임베딩하고 ollama요청해서 답변받아오는 부분

    # 임베딩 embed

    def embed(self, text: str) -> List[float]: #리턴타입!!
        #ollama에 주소로 임베딩해달라고 요청합시다.
        url = self.ollama_base_url + '/api/embeddings'
        resp = requests.post(url, json={'model': self.embed_model, 'prompt': text}, timeout=120)
        print(resp.json()) #dict형태로 만들어서 프린트.
        data = resp.json() #{"embedding" : [0.1232, 0.234324]}
        return data['embedding']

    # 답생성 generate
    def generate(self, prompt: str) -> str:
        url = self.ollama_base_url + '/api/generate'
        payload = {
            "model": self.gen_model,  # gemma3:1b
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0, #네가 찾은 것 중에서 아주 정확한 것만!!
                "num_predict": 64,
                # 짧게, 한글은 한글자에 2바이트임, 30글자 정도, 문장 2-3개
                # 환경에 따라 지원되는 옵션이 다를 수 있음
                # "repeat_penalty": 1.1,
            }
        }
        r = requests.post(url, json=payload, timeout=120)
        print(r.json())
        data = r.json()
        return data['response']


    def ingest_texts(self, texts: List[str], source: str = "manual") -> int:
        if not texts:
            return 0

        for i, t in enumerate(texts): ## collection이 table이면  ids,documents,embeddings,metadatas는 컬럼인가요?
            self.collection.add(
                ids=[str(uuid.uuid4())],
                documents=[t],
                embeddings=[self.embed(t)],
                metadatas=[{"chunk": i, "source": source}],
                )
        return len(texts)


    def count(self)->int:
        return self.collection.count()

    def get_collection(self):
        results=self.collection.get(include=['documents','metadatas'],limit=10)
        return results
    @staticmethod
    def chunk_text(text: str, max_chars: int = 1200, overlap_chars: int = 150) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        chunks = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + max_chars, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end == n:
                break

            start = max(0, end - overlap_chars)

        return chunks




###############
# 3. chuck만드는 부분

    # 통으로 읽은 text를 작게 자르자(chuck, 조각)
    # pdf를 읽어서 text로 만들자.

    # collection에 몇 개 들어있는지 확인하는 함수

###############
# 4. 크로마 db에 적재하는 부분

    # 텍스트를 크로마db에 적재하자.(ingest)

###############
# 5. 질문하고 답변 만들어오는 부분
    # 질의하고 답변을 만드는 것과 관련된 함수 정의.


if __name__ == '__main__':
    rag = ChromaRAG() # def __init__() 호출됨.
    # print(rag) # def __str__()호출됨.
    # rag.embed(text='hello')
    # rag.embed(text='world')
    # print(rag.generate(prompt='마라탕'))
    text='앞서 MC몽은 보도 내용은 모두 사실이 아니라고 반박하며 차 회장의 작은아버지인 차준영씨가 자신에게 빅플래닛메이드의 지분을 내놓으라며 협박했고 이 과정에서 차 회장과 친구인 자신의 관계를 조작한 대화를 넘겼다고 주장했다.아울러 차 회장에 대해 "가정에 충실한 친구"라며 "그 친구와 저는 늘 아티스트와 함께 만났다. 기사가 나갔을 때 BPM, 원헌드레드 아티스트가 모두 웃었을 것"이라고 반박했다.원헌드레드 측 역시 공식입장을 통해 "기사 내용과 카톡 대화는 모두 사실이 아니었다. 이는 MC몽이 차가원 회장의 친인척인 차준영씨로부터 협박받고 조작해서 보낸 것"이라며 "당시 차씨는 빅플래닛메이드 경영권을 뺐기 위해 MC몽에게 강제적으로 주식을 매도하게 협박했으며 이 과정에서 MC몽의 조작된 카톡이 전달된 것"이라고 설명했다.'
    result=ChromaRAG.chunk_text(text)
    print(result)
    result2=rag.embed(text=result[0])
    print('result2',result2)
    print('rag.generate',rag.generate(prompt=text))
    result3=rag.ingest_texts(result)
    print('result3',result3)
    print(rag.count())

    print(rag.get_collection())

    text1 = "웹툰 작가 겸 방송인 기안84의 과거 발언이 최근 코미디언 박나래를 둘러싼 이른바 주사 이모 논란과 맞물리며 주목받고 있다.25일 연예계에 따르면 최근 여러 온라인 커뮤니티와 SNS에서는 약 4개월 전 공개된 기안84의 유튜브 채널 '인생84' 영상 일부가 다시 확산되고 있다.문제의 장면은 배우 이세희가 출연한 편으로, 당시 기안84가 이세희의 자택을 방문해 대화를 나누는 과정에서 나온 한 발언이다.영상에서 이세희는 집에 놀러 오면 직접 해준다며 평소 즐겨 한다는 피부 관리를 소개했고, 중고 플랫폼을 통해 약 200만 원에 구매했다는 고주파 피부 관리 기계를 꺼내 기안84에게 직접 시연에 나섰다."
    result = ChromaRAG.chunk_text(text)
    print(result)
    result2 = rag.embed(text=result[0])
    print('result2', result2)
    print('rag.generate', rag.generate(prompt=text))
    result3 = rag.ingest_texts(result)
    print('result3', result3)
    print(rag.count())

    print(rag.get_collection())
    text2 = "두 사람은 피부 관리에 앞서 책과 일상적인 화제에 대해 대화를 나누던 중이었고, 이 과정에서 기안84는 고주파 기계를 바라보며 '앞에서는 되게 아름다운 이야기를 하다가 갑자기 이걸 꺼내니까 가정 방문 야매 치료사 같다고 농담을 던졌다.당시 발언은 가벼운 농담으로 받아들여지며 별다른 논란 없이 지나갔지만 최근 박나래가 지인으로 알려진 이른바 주사 이모'에게 받은 불법 의료 시술 사건이 공론화되면서 해당 방송 장면이 다시 소환됐다.누리꾼들은 지금 보니 진짜 의미심장하다, 우연이라고 볼 수 없는 묘한 발언, 괜히 다시 떠오르고 있는 장면이 아니다. 이미 뭔고 알고 있었던 것 이라는 반응을 보인다.한편 박나래는 최근 '주사 이모' 의혹과 더불어 전 매니저들과의 갈등이 수면 위로 드러나며 여러 논란에 휩싸였다. 박나래는 불법 의료 시술 의혹과 전 매니저에 대한 갑질 의혹 등이 잇따라 제기되자 지난 8일 방송 활동을 모두 중단했다. 현재는 법적 절차에 따라 대응하겠다는 방침을 유지하고 있다."
    result = ChromaRAG.chunk_text(text)
    print(result)
    result2 = rag.embed(text=result[0])
    print('result2', result2)
    print('rag.generate', rag.generate(prompt=text))
    result3 = rag.ingest_texts(result)
    print('result3', result3)
    print(rag.count())

    print(rag.get_collection())

    text3 = "이날 첫인상 투표 결과 용담과 국화는 1표, 백합은 3표였다. 표를 받지 못한 장미와 튤립. 장미는 '창피하고, 서운하기도 했다. 분발해야겠단 생각을 했다' 했고, 튤립은 '친구들이 엄청 놀리겠다 싶었다. 엄마한테 뭐라 하지. (엄마가) 놀리기도 하고, 진심으로 실망할 것 같다고 걱정했다. 5분 대화 후 백합은 22기 상철로 마음이 바뀌었다.28기 영수는 자기 소개한 지 얼마 안 됐다면서 84년생이며, 스타트업 창업가라고 밝혔다. 영수는 돌싱인데, 2년 정도 됐다. 아이는 없다라며 지금까지 거짓말치는 인생은 살지 않았다라며 '나는 솔로' 방영 후 시청자들의 반응을 의식했다. 영수는 방송 보면서 부족한 점을 많이 느꼈다. 선입견이 있다면 선입견을 내려놓고 대화를 나누었으면 좋겠다고 전했다."
    result = ChromaRAG.chunk_text(text)#==>chunk하고
    print(result)
    result2 = rag.embed(text=result[0])
    print('result2', result2)
    print('rag.generate', rag.generate(prompt=text))
    result3 = rag.ingest_texts(result)#==>적재하자
    print('result3', result3)
    print(rag.count())

    print(rag.get_collection())

    list=[text1,text2,text3]
    for x in list:
        result=ChromaRAG.chunk_text(x)
        print(result)
        result3 = rag.ingest_texts(result)  # ==>적재하자
        print('result3', result3)
        print('============================================================')