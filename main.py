import ollama
import asyncio
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
# from ollama_client import stream_generate
from redis.asyncio import Redis  # redis-py의 async 클라이언트
import json
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
# .env 파일 로드

# from fastapi.middleware.cors import CORSMiddleware
# from transformers import pipeline


class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    message: str


load_dotenv()
app = FastAPI()
# -----------------------
# CORS 설정
# -----------------------
# .env에서 읽어서 리스트 형태로 변환
origins = os.getenv("CORS_ORIGINS", "").split(",")


# Redis 설정 (로컬 개발 기준, 프로덕션에서는 환경변수로 관리)
MODEL = "gemma3:1b"
OLLAMA_BASE_URL = "http://localhost:11434"
REDIS_URL = "redis://localhost:6379"
# 앱 상태에 Redis 클라이언트 저장
app.state.redis = None  # 아직 연결안됨. fastapi시작할 때 redis도 연결해두려고 함.

# Static 파일 설정 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Templates 설정
templates = Jinja2Templates(directory="templates")

# CORS 설정 (클라이언트 도메인 허용, 개발 시 "*"로 테스트)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한, 예: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 사용할 모델 이름 (미리 ollama pull <model>로 다운로드 필요, 예: ollama pull llama3.2)
# MODEL = "llama3.2"


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})


@app.get("/health")
async def health_check():
    """FastAPI와 Ollama의 health 상태를 확인하는 엔드포인트"""
    try:
        # Ollama health check
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)

        if response.status_code == 200:
            ollama_status = "healthy"
            message = "fastapi & ollama 제대로 동작중"
        else:
            ollama_status = "unhealthy"
            message = f"Ollama returned status code: {response.status_code}"

    except httpx.ConnectError:
        ollama_status = "연결불가"
        message = "Ollama 연결할 수 없음."
    except httpx.TimeoutException:
        ollama_status = "타임아웃"
        message = "Ollama 타임 아웃"
    except Exception as e:
        ollama_status = "error"
        message = f"Error checking Ollama: {str(e)}"

    return HealthResponse(
        status="ok",
        ollama_status=ollama_status,
        message=message
    )


# 앱 시작 시 모델 미리 로드 (preload)
@app.on_event("startup")
async def preload_model():
    try:
        # 빈 프롬프트로 모델 로드 + 영구 유지
        await ollama.AsyncClient().generate(
            model=MODEL,
            prompt=" ",  # 빈 프롬프트 (또는 "preload" 같은 더미 텍스트)
            keep_alive=-1  # -1: 영구적으로 메모리에 유지
        )
        print(f"{MODEL} 모델이 미리 로드되었습니다. (메모리에 영구 유지)")

        app.state.redis = Redis.from_url(url=REDIS_URL, decode_responses=True)
        # decode_responses=True --> 바이트스트림으로 도착한 데이터 utf-8로 자동으로 변환

        print(f"{REDIS_URL}로 Redis서버 미리 연결됨.")

    except Exception as e:
        print(f"모델 preload 실패 또는 Redis연결 실패 : {e}")


# fastapi서버가 종료(재부팅)되었을 때 자동 호출됨.
@app.on_event("shutdown")
async def shutdown_event():
    if app.state.redis:
        await app.state.redis.close()
        print("redis 연결 종료됨.....")


# 일반 generate 엔드포인트 (스트리밍 없이 전체 응답)
@app.get("/chat")
async def generate(word: str, request: Request):
    try:
        response = await ollama.AsyncClient().generate(
            model=MODEL,
            prompt=word,
            options={"temperature": 1},
            keep_alive=-1  # 필요 시 후속 요청에서도 유지
        )
        return templates.TemplateResponse("chat.html",
                                          context={"request": request,
                                                   "result": response["response"]
                                                   })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 스트리밍 엔드포인트 (실시간 토큰 반환, 더 빠른 체감)
from fastapi.responses import StreamingResponse


@app.get("/stream")
async def stream(word: str):
    return StreamingResponse(stream_generate(word), media_type="text/event-stream")


async def stream_generate(prompt: str):
    stream = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        stream=True,
        keep_alive=-1
    )
    async for part in stream:
        # "이 값을 내보내고, 여기서 잠깐 멈춰. 다음에 다시 불러주면 이어서 할게!"
        # ollama로 부터 받은 조각마다 보내..
        yield part["response"]


@app.get("/ollama-test")
def ollama_test(request: Request):
    return templates.TemplateResponse("ollama-test.html", context={"request": request})


## 파라메터 전달용 class를 만들자.
## 파라메터 이름 똑같은거 자동으로 변수에 들어감.
## 다른 옵션값들 설정 가능
## BaseModel이라는 클래스를 상속받아서 만들어야 자동으로
## 이런 처리들을 해줌.

class SummarizeRequest(BaseModel):
    ## BaseModel(변수+함수) + 내가 추가한 변수
    text: str
    max_length: int = 200


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    # http://localhost:11434/api/generate, json=payload
    # post방식으로 http요청을 해줌.
    prompt = f"{request.text}를 {request.max_length}자로 요약해주세요."
    print(prompt)
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------")
    print(response)  # dict
    return {'summary': response["response"].strip()}


class NamesRequest(BaseModel):
    #axios.post로 전달될 때 키와 이름이 같아야 한다.
    category:str='카페'
    gender:str="중성"
    vibe:str='귀여운'
    count:int=3
@app.post('/create_name')
async def translate(request: NamesRequest):
    print('request',request.category)
    #http://localhost:11434/api/generate,json=payload를 generatoe()함수가 대신 해준다.
    #post 방식으로 http요청을 해줌.
    prompt=f"""{request.category}의 이름을 
                {request.gender}느낌으로 
                {request.count}개만 추천해주세요.
            """
    print(prompt)
    response=await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print('------------------------')
    print(response['response'])
    return {'summary':response['response'].strip()}

# 추가 엔드포인트
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # 간단히 세션 구분


# 메모리 기반 간단한 대화 히스토리 저장 (프로덕션에서는 Redis 등 사용)
chat_histories = {}


########### redis연결전
# @app.post("/chat")
async def chat(request: ChatRequest):
    history = chat_histories.get(request.session_id, [])
    history.append({"role": "user", "content": request.message + ", 200글자 이내로 핵심만 답을 줘."})
    # 나는 user, ai는 assistant

    response = await ollama.AsyncClient().chat(
        model=MODEL,
        messages=history,
        keep_alive=-1
    )

    print("-----------------")
    print(response)
    # message = Message(role='assistant',
    #                   content='싱가포르는 매력적인 도시로, 다양한 경험을 제공하는 매혹적인 나라입니다. 싱가포르 여행에 대한 유용한 정보들을 정리해 드릴게요.\n\n**1. 여행 준비**\n\n*   **비자:** 한국인은 90일까지 무비자 체류 가능합니다
    ai_message = response["message"]["content"]
    history.append({"role": "assistant", "content": ai_message})
    chat_histories[request.session_id] = history[-10:]  # 최근 10턴 유지

    print("chat_histories>> ", chat_histories)
    # chat_histories >> {'default': [{'role': 'user', 'content': '싱가폴 여행정보, 200글자 이내로 핵심만 답을 줘.'},
    # {'role': 'assistant', 'content': '싱가포르는 다채로운 문화와 현대적인 도시, 그리고 맛있는 음식으로 유명합니다. \n\n*   **교통:** 대중교통 시스템이 매우 잘 갖춰져 있어 편리하게 이동할 수 있습니다.\n*   **관광 명소:** 마리나 베이 푹, 센토사 섬, 칠리 궁전 등 다양한 명소가 있습니다.\n*   **특징:** 24시간 영업하는 식료품점, 다양한 길거리 음식, 럭셔리한 쇼핑 등 독특한 경험을 할 수 있습니다.\n\n**여행 준비:** 미리 항공권과 숙소를 예약하고, 환전은 한국 원화로 하는 것이 좋습니다.'}]}
    return {"response": ai_message}


# redis에 키를 chat_history:로그인id로 만들어줄 예정.
# id가 apple인 경우 키는 chat_history:apple
# id가 default인 경우 키는 chat_history:default
# chat_history:apple, chat_history:default는 키이므로 unique해야함.
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # 로그인한 아이디


# 기존 /chat 엔드포인트 수정
########### redis연결후
# @app.post("/chat")
# async def chat(request: ChatRequest):
#     print(f"서버로 전달된 값은 {request.message}, {request.session_id}")
#     return {"response": "ok"}

class ChatRequest(BaseModel):
    message:str
    session_id:str='default'
    #redis에 키를 chat_history:로그인id로 만들어줄 예정

@app.post('/chat')
async def chat(request: ChatRequest):
    print('request',request.message,request.session_id)
    history=[]
    #http://localhost:11434/api/generate,json=payload를 generatoe()함수가 대신 해준다.
    #post 방식으로 http요청을 해줌.
    user_message=request.message
    history.append({"role":"user","content":user_message})
    # print(user_message)
    response=await ollama.AsyncClient().chat(
        model=MODEL,
        messages=history,
        keep_alive=-1
    )
    print('------------------------')
    print(response['message'])
    ai_message = response['message']['content']
    history.append({"role":"assistant","content":ai_message})
    print('history',history)

    #레디스에 넣읍시다.
    session_key="chat_history:"+request.session_id
    redis=app.state.redis
    #레디스에는 json으로 넣어줘야 하므로 dict를 json.dumps함수로 json으로 바꿔준다.
    if history:
            await redis.rpush(session_key, *[json.dumps(one) for one in history])

    return {'response':response['message']['content'].strip()}

@app.get('/chat-history/{session_id}')
async def get_chat_history(session_id: str):
    redis=app.state.redis
    if not redis:
        raise HTTPException(status_code=500, detail="레디스 연결 안됨.")
    session_key=f"chat_history:{session_id}"
    history_json=await redis.lrange(session_key,0,-1)
    history=[json.loads(msg) for msg in history_json]
    # print ('history',history)
    return history

class ProposesRequest(BaseModel):
    story:str
    jenre:str
    actor:str
    count:int
    max_length:int=15
@app.post('/propose')
async def propose(request: ProposesRequest):
    print('request',request.story,request.jenre,request.actor)
    #http://localhost:11434/api/generate,json=payload를 generatoe()함수가 대신 해준다.
    #post 방식으로 http요청을 해줌.
    prompt=f"""내용이 {request.story}인 
                {request.jenre} 스타일로
                {request.actor}가 주인공인
{request.max_length}글자 이내의 영화 제목을 {request.count}개 만들어 주세요.
            """
    print(prompt)
    response=await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print('------------------------')
    print(response['response'])
    return {'summary':response['response'].strip()}

class ChatRequest(BaseModel):
    message:str
    session_id:str='default'
    #redis에 키를 chat_history:로그인id로 만들어줄 예정

class PoemsRequest(BaseModel):
    topic:str
    style:str
@app.post('/ai/poem')
async def translate(request: PoemsRequest):
    print('request',request.topic,request.style)
    #http://localhost:11434/api/generate,json=payload를 generatoe()함수가 대신 해준다.
    #post 방식으로 http요청을 해줌.
    prompt=f"""{request.topic}을 주제로 
                {request.style}스타일로 시를 지어 주세요.
            """
    print(prompt)
    response=await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print('------------------------')
    print(response['response'])
    return {'summary':response['response'].strip()}

class TranslateRequest(BaseModel):
    text:str
@app.post('/translate')
async def translate(request: TranslateRequest):
    #http://localhost:11434/api/generate,json=payload를 generatoe()함수가 대신 해준다.
    #post 방식으로 http요청을 해줌.
    prompt=f"{request.text}를 자연스러운 한국어로 번역해주세요"
    print(prompt)
    response=await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print('------------------------')
    print(response['response'])
    return {'summary':response['response'].strip()}