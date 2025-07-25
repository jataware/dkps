import asyncio
import numpy as np
from tqdm import trange
from google import genai
from tqdm.asyncio import tqdm

from .cache import disk_cache

@disk_cache(cache_dir='./.cache/embed/google', verbose=False, ignore_fields=['client'])
async def _aembed_google_chunk(chunk_id, client, chunk, model):
    chunk_response = await client.aio.models.embed_content(
        model    = model,
        contents = chunk,
    )
    return chunk_id, np.array([xx.values for xx in chunk_response.embeddings])

@disk_cache(cache_dir='./.cache/embed/google', verbose=False)
async def aembed_google(input_strs, chunk_size=100, model='gemini-embedding-001', max_concurrency=5):
    # TODO - add error handling
    
    client = genai.Client()
    
    sem = asyncio.Semaphore(max_concurrency)
    async def _fn(chunk_id, chunk):
        async with sem:
            return await _aembed_google_chunk(chunk_id, client, chunk, model)
    
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


def embed_google(*args, **kwargs):
    return asyncio.run(aembed_google(*args, **kwargs))

def embed_nomic(input_strs):
    raise NotImplementedError('Not implemented')

# def embed_other_stuff(input_strs):
#     raise NotImplementedError('Not implemented')
