from fastapi import APIRouter,HTTPException
from typing import List,Dict
import json
from fastapi import Depends
from fastapi.responses import JSONResponse
import httpx
import uuid

router = APIRouter()


async def fetch_message(url:str) -> List[Dict]:
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status() 
            return response.json()
    
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=f"Error fetching data: {exc.response.text}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=500, detail=f"An error occurred while requesting {exc.request.url!r}.")


@router.post("/fetch_response")
async def fetch_message_with_sources():
    url = "https://devapi.beyondchats.com/api/get_message_with_sources"
    
    try:
        data = await fetch_message(url)
        data = data["data"]["data"]
        
        id = []
        source_lst = []
        response_lst = []
        citations_lst = []
        for item in data:
            id.append(item["id"])
            if item["source"] and item["response"]:
                response_lst.append(item["response"])
                source_lst.append(item["source"])
                citations = []
                for source in item["source"]:
                    if source["link"] != "":
                        citations.append({
                            "id": source["id"],
                            "link": source["link"]
                        })
                citations_lst.append(citations)

        return JSONResponse(content={"id":id,"response":response_lst,"source":source_lst,"citations":citations_lst}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

