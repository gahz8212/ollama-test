from fnmatch import translate

import ollama
import httpx
import redis
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from pydantic.v1 import ConstrainedSet
from redis.asyncio import Redis

from ollama_client import ollama_client


#응답을 JSON으로 해주는 부품(class)을 만들자.]
# BaseModel의 모든 변수+함수를 다 가지고 와서 확장해라.(상속)
# Taxt(Car) : Car가 가지고 있는 모든 변수+함수를 다 가지고 와서 확장해라.
# Truck(Car)

class HealthResponse(BaseModel):
    status : str
    ollama_status : str
    message: str

app = FastAPI()

MODEL="gemma3:1b"
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama 기본 포트
REDIS_URL="redis://localhost:6379"

app.state.redis=None

# Static 파일 설정 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Templates 설정
templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_root(request : Request):
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

# 앱 시작시 모델 미리 로드(preload)
@app.on_event("startup")
async def preload_model():
    try:
        #빈 프롬프트로 모델 로드+영구유지
        await ollama.AsyncClient().generate(
            model=MODEL,
            prompt="",
            keep_alive=-1#영구 유지
        )
        print(f"{MODEL} 모델이 미리 로드되었습니다.")
        app.state.redis = redis.from_url(url=REDIS_URL,decode_responses=True)
        #decode_response=True --> 바이트스트림으로 도착한 데이터 utf8로 자동 변환
        print(f"{REDIS_URL}로 Redis 서버 미리 연결됨.")

    except Exception as e:
        print(f"모델 preload 실패 또는 Redis연결 실패: {e}")


# @app.on_event("shutdown")
# async def shutdown_event():
#     if app.state.redis:
#         await app.state.redis.close()main.py

@app.on_event("shutdown")
async def shutdown_event():
    print("==================")
    print(app.state.redis)
    if app.state.redis:
        await app.state.redis.close()
        print('redis 연결 종료됨...')

# 일반 generate 엔드포인트 (스트리밍 없이 전체 응답)


# @app.get("/chat")
# async def generate(word : str, request : Request):
#     print('word',word)
#     try:
#         response = await ollama.AsyncClient().generate(
#             model=MODEL,
#             prompt=word,
#             options={"temperature": 1},
#             keep_alive=-1  # 필요 시 후속 요청에서도 유지
#         )
#         return templates.TemplateResponse("chat.html",
#                                       context={"request": request,
#                                                "result" : response["response"]
#                                                })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.get('/stream')
async def stream(word:str):
    return StreamingResponse(stream_generate(word),media_type="text/event-stream")

async def stream_generate(prompt: str):
    stream = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        stream=True,##꼭!
        keep_alive=-1
    )
    async for part in stream:
        # "이 값을 내보내고, 여기서 잠깐 멈춰. 다음에 다시 불러주면 이어서 할게!"
        # ollama로 부터 받은 조각마다 보내..
        yield part["response"]#리덕스 사가의 generate함수와 비슷한 것 같다.



@app.get('/ollama-test')
async def stream(request: Request):
    return templates.TemplateResponse("ollama-test.html",
                                      context={"request": request, })

##파라메터 전달용 class 만들자
##파라메터 이름 똑같은거 자동으로 변수에 들어감.
##다른 옵션값들 설정 가능
##BaseModel이라는 클래스를 상속받아서 만들어야 자동으로 처리해 줌.
class SummarizeRequest(BaseModel):
    ##BaseModel(변수+함수)+내가 추가한 변수
    text:str
    max_length:int=200

@app.post('/summarize')
async def summarize(request: SummarizeRequest):
    #http://localhost:11434/api/generate,json=payload를 generatoe()함수가 대신 해준다.
    #post 방식으로 http요청을 해줌.
    prompt=f"{request.text}를{request.max_length}자로 요약해주세요"
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

class PoemsRequest(BaseModel):
    topic:str
    style:str
@app.post('/poem')
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

@app.post('/chat')
async def chat(request: ChatRequest):
    print('request',request.message,request.session_id)
    #http://localhost:11434/api/generate,json=payload를 generatoe()함수가 대신 해준다.
    #post 방식으로 http요청을 해줌.
    prompt=request.message
    # print(prompt)
    response=await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print('------------------------')
    print(response['response'])
    return {'summary':response['response'].strip()}


if __name__ == '__main__':
    uvicorn.run("main:app", host='127.0.0.1', port=8001, reload=True)