from fastapi import APIRouter
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
from api.schemas.chat_schema import ChatRequest
from fastapi.responses import JSONResponse
import asyncio
from api.engine import rag_engine

router = APIRouter()
rag_comp = rag_engine.RagEngine()



@asynccontextmanager
async def load_model() -> AsyncGenerator[rag_engine.RagEngine,None]:
    try:
        yield rag_comp
    
    finally:
        pass


from fastapi import status
from fastapi import HTTPException

async def generate_word(prompt: str):
    try:
        async with load_model() as model:
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, model.create_context_processor, prompt)
            gen_word = await asyncio.wait_for(future, 12000)
            for word in gen_word.response_gen:
                await asyncio.sleep(0.01)
                yield word

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail={"Took too long to respond"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server error"
        ) from e

@router.post("/chat/api/instruct_resp", tags=["chat"])
async def generate(chat_request: ChatRequest):
    try:
        user_prompt = chat_request.prompt
        response = StreamingResponse(
            generate_word(user_prompt),
            status_code=200,
            media_type="text/plain"
        )
        return response
    except HTTPException as e:
        return JSONResponse(
            content={"error": e.detail},
            status_code=e.status_code
        )
    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=200,
            content={"error": str(e)}
        )

