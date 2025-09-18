import os
import asyncio
import numpy as np
from tqdm import trange
from tqdm.asyncio import tqdm
import httpx

try:
    from google import genai
    from google.genai import types
    from google.genai.types import HttpOptions
except:
    print('dkps.embed: unable to load google-genai')

try:
    from jlai.embed.tei import embed_dataset as embed_jlai_tei
except:
    print('dkps.embed: unable to load jlai')

from .cache import disk_cache

# --
# Provider-specific chunk functions

# Google
@disk_cache(cache_dir='./.cache/embed/google', verbose=False, ignore_fields=['client'])
async def _aembed_google_chunk(chunk_id, client, chunk, model):
    chunk_response = await client.aio.models.embed_content(
        model    = model,
        contents = chunk,
    )
    return chunk_id, np.array([xx.values for xx in chunk_response.embeddings])

# Jina
class JinaClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise Exception("JINA_API_KEY is not set")
        
        self.headers = {
            "Accept"          : "application/json",
            "Authorization"   : f"Bearer {self.api_key}",
            "Content-Type"    : "application/json",
        }
    
    async def embed(self, input_data, model):
        async with httpx.AsyncClient(timeout=None, headers=self.headers) as client:
            res = await client.post("https://api.jina.ai/v1/embeddings", json={
                "model" : model,
                "task"  : "text-matching",
                "input" : input_data,
            })
            res.raise_for_status()
            return res.json()

@disk_cache(cache_dir="./.cache/embed/jina", verbose=False, ignore_fields=['client'])
async def _aembed_jina_chunk(chunk_id, client, chunk, model):
    data = await client.embed(chunk, model)
    return chunk_id, np.array([item['embedding'] for item in data['data']])


# Generic API embedding functions
async def _aembed_api(provider, input_strs, chunk_size=50, max_concurrency=5, model=None):
    """Generic async embedding function that handles chunking and concurrency"""
    assert isinstance(input_strs, list), 'input_strs must be a list'
    
    if provider == 'google':
        client = genai.Client(
            api_key=os.getenv('GEMINI_API_KEY'),
            http_options=HttpOptions(
                api_version='v1beta',
                timeout=10 * 1000, # 10 seconds
            ),
        )
        _aembed_chunk = _aembed_google_chunk
        if model is None:
            model = 'gemini-embedding-001'
        
        
    elif provider == 'jina':
        client = JinaClient()
        _aembed_chunk = _aembed_jina_chunk
        if model is None:
            model = 'jina-embeddings-v3'
    
    
    else:
        raise Exception


    sem = asyncio.Semaphore(max_concurrency)
    async def _fn(chunk_id, chunk):
        async with sem:
            return await _aembed_chunk(chunk_id, client, chunk, model)
    
    chunks = [input_strs[i:i+chunk_size] for i in range(0, len(input_strs), chunk_size)]
    
    tasks = []
    for chunk_id, chunk in enumerate(chunks):
        task = _fn(chunk_id, chunk)
        tasks.append(task)
    
    out = [None] * len(chunks)
    for task in tqdm(asyncio.as_completed(tasks), desc="Embedding chunks", total=len(tasks)):
        chunk_id, embedding = await task
        out[chunk_id] = embedding
    
    return np.concatenate(out)

# jlai
@disk_cache(cache_dir="./.cache/embed/jlai_tei", verbose=False, ignore_fields=['client'])
def _embed_jlai_tei(input_strs, model='nomic-ai/nomic-embed-text-v1', **kwargs):
    assert isinstance(input_strs, list), 'input_strs must be a list'
    
    return embed_jlai_tei(input_strs, model_id=model)


# Synchronous wrapper functions
def embed_api(provider, *args, **kwargs):
    if provider == 'jlai_tei':
        return _embed_jlai_tei(*args, **kwargs)
    else:
        return asyncio.run(_aembed_api(provider, *args, **kwargs))
