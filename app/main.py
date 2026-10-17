import base64
import io
import json
import queue
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from uuid import uuid4
from app.config import settings
from app.agent import get_graph, close_graph

ROOT=Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app):
    if settings.app_env == 'production' and not settings.api_token:
        raise RuntimeError('API_TOKEN must be set when APP_ENV=production')
    get_graph()  # initialize the agent + SQLite checkpointer once at startup
    yield
    close_graph()


app=FastAPI(title='Medical Agent Educational API',version='1.0.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=settings.allowed_origins.split(','),allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['Authorization','Content-Type'])
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')


class ChatRequest(BaseModel):
    message: str = Field(min_length=1,max_length=4000)
    history: list[dict[str,str]] = Field(default_factory=list,max_length=10)
    image_data: str | None = None

def auth(authorization: str | None):
    if settings.api_token and not secrets.compare_digest(authorization or '',f'Bearer {settings.api_token}'):
        raise HTTPException(status_code=401,detail='Invalid API token')


def inspect_image(data):
    if not data:
        return ''
    try:
        header,payload=data.split(',',1)
        mime=header.split(';')[0].split(':')[1]
        raw=base64.b64decode(payload,validate=True)
        if mime not in ('image/jpeg','image/png','image/webp') or len(raw)>5_000_000:
            raise ValueError('Only JPEG/PNG/WebP under 5MB supported')
    except (ValueError,IndexError) as exc:
        raise HTTPException(status_code=400,detail=f'Invalid image: {exc}')
    client=OpenAI(api_key=settings.openai_api_key,timeout=45)
    result=client.chat.completions.create(model=settings.openai_vision_model,messages=[{'role':'user','content':[{'type':'text','text':'Describe only objectively visible medical-relevant features. Do not diagnose, estimate severity, or assert absence of illness. If image unclear, say so.'},{'type':'image_url','image_url':{'url':data}}]}],max_tokens=300)
    return result.choices[0].message.content or ''


def _sources_payload(state):
    kb = state.get('kb', [])
    web = state.get('web', [])
    return (
        [{'type': 'pdf', 'page': x['page'], 'score': x['score'], 'text': x['text'][:600]} for x in kb]
        + [{'type': 'web', 'url': x['url'], 'title': x['title']} for x in web]
    )


def _tools_payload(state, observation):
    return (
        (['pinecone'] if state.get('kb') else [])
        + (['tavily'] if state.get('route') == 'web' and settings.tavily_api_key else [])
        + (['vision'] if observation else [])
    )


def _confidence_payload(state):
    """Honest evidence signals for the UI.

    low_confidence is True when a medical answer was generated with little to
    ground it: no web results and KB missing or too weak to answer directly.
    Guardrail fast-paths (emergency/greeting/off-topic) are not graded.
    """
    kb = state.get('kb', [])
    top = kb[0]['score'] if kb else None
    low = (
        state.get('intent') == 'medical'
        and not state.get('web')
        and (not kb or kb[0]['score'] < settings.web_fallback_score)
    )
    return {'kb_top_score': top, 'route': state.get('route', ''), 'low_confidence': bool(low)}


@app.get('/',response_class=HTMLResponse)
def home():
    return (ROOT/'templates'/'index.html').read_text(encoding='utf-8')

@app.get('/api/health')
def health():
    return {'status':'ok','configured':bool(settings.openai_api_key and settings.pinecone_api_key),'tavily':bool(settings.tavily_api_key)}



@app.post('/api/chat')
def chat(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(default=None, alias='X-Conversation-ID')
):
    auth(authorization)

    if not settings.openai_api_key or not settings.pinecone_api_key:
        raise HTTPException(503,'Set OPENAI_API_KEY and PINECONE_API_KEY in .env')

    history=[x for x in payload.history if x.get('role') in ('user','assistant') and isinstance(x.get('content'),str)][:10]

    # SQLite persistent memory: reuse the same conversation thread
    thread_id = conversation_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        observation=inspect_image(payload.image_data)

        state=get_graph().invoke(
            {
                'question':payload.message,
                'history':history,
                'image_description':observation,
                'kb':[],
                'web':[],
                'answer':'',
                'route':'',
                'intent':'',
                'has_image':bool(payload.image_data),
                'validation_flags':[]
            },
            config=config
        )

        return {
            'answer':state['answer'],
            'intent':state.get('intent',''),
            'sources': _sources_payload(state),
            'tools': _tools_payload(state, observation),
            'conversation_id':thread_id,
            **_confidence_payload(state),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print('Chat error:',repr(exc),flush=True)
        raise HTTPException(502,'AI service failed. Check server logs, credentials, Pinecone index, and connectivity.')


@app.post('/api/chat/stream')
def chat_stream(
    payload: ChatRequest,
    authorization: str | None = Header(default=None),
    conversation_id: str | None = Header(default=None, alias='X-Conversation-ID')
):
    """Server-sent events stream of the answer.

    Events: {"type":"token","text":"..."} as the answer node generates,
    then {"type":"done", ...full payload...}. Guardrail fast-paths
    (emergency/greeting/off-topic) send no tokens — just the done event.
    """
    auth(authorization)

    if not settings.openai_api_key or not settings.pinecone_api_key:
        raise HTTPException(503,'Set OPENAI_API_KEY and PINECONE_API_KEY in .env')

    history=[x for x in payload.history if x.get('role') in ('user','assistant') and isinstance(x.get('content'),str)][:10]
    thread_id = conversation_id or str(uuid4())
    token_q: queue.Queue = queue.Queue()
    config = {"configurable": {"thread_id": thread_id, "token_queue": token_q}}
    observation=inspect_image(payload.image_data)

    input_state = {
        'question':payload.message,
        'history':history,
        'image_description':observation,
        'kb':[],
        'web':[],
        'answer':'',
        'route':'',
        'intent':'',
        'has_image':bool(payload.image_data),
        'validation_flags':[]
    }
    result: dict = {}

    def run():
        try:
            result['state'] = get_graph().invoke(input_state, config=config)
        except Exception as exc:
            print('Chat stream error:',repr(exc),flush=True)
            token_q.put({'type':'error','detail':'AI service failed. Check server logs, credentials, Pinecone index, and connectivity.'})
        finally:
            token_q.put({'type':'end'})

    threading.Thread(target=run, daemon=True).start()

    def gen():
        while True:
            item = token_q.get()
            if item.get('type') == 'end':
                break
            yield f"data: {json.dumps(item)}\n\n"
        st = result.get('state') or {}
        done_payload = {
            'type': 'done',
            'answer': st.get('answer', ''),
            'intent': st.get('intent', ''),
            'sources': _sources_payload(st),
            'tools': _tools_payload(st, observation),
            'conversation_id': thread_id,
        }
        done_payload.update(_confidence_payload(st))
        yield 'data: ' + json.dumps(done_payload) + '\n\n'

    return StreamingResponse(gen(), media_type='text/event-stream')


@app.post('/api/transcribe')
async def transcribe(file:UploadFile=File(...),authorization:str|None=Header(default=None)):
    auth(authorization)
    data=await file.read(10_000_001)
    if len(data)>10_000_000:
        raise HTTPException(413,'Audio exceeds 10 MB')
    audio=io.BytesIO(data); audio.name='voice.webm'
    try:
        result=OpenAI(api_key=settings.openai_api_key).audio.transcriptions.create(model='whisper-1',file=audio)
        return {'text':result.text}
    except Exception:
        raise HTTPException(502,'Speech transcription unavailable')

@app.post('/api/speak')
def speak(payload:dict,authorization:str|None=Header(default=None)):
    auth(authorization)
    text=str(payload.get('text',''))[:3000]
    if not text:
        raise HTTPException(400,'Missing text')
    try:
        speech=OpenAI(api_key=settings.openai_api_key).audio.speech.create(model='tts-1',voice='alloy',input=text)
        return Response(content=speech.content,media_type='audio/mpeg')
    except Exception:
        raise HTTPException(502,'Speech synthesis unavailable')