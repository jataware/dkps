import numpy as np
from tqdm import trange
from google import genai

def embed_google(input_strs, model='gemini-embedding-001'):
    # TODO - make this async
    
    client = genai.Client()

    all_embeddings = []
    for offset in trange(0, len(input_strs), 100):
        chunk = [str(xx) for xx in input_strs[offset:offset+100]]
        chunk_response = client.models.embed_content(
            model    = model,
            contents = chunk,
        )
        all_embeddings.extend([xx.values for xx in chunk_response.embeddings])

    return np.array(all_embeddings)

def embed_nomic(input_strs):
    raise NotImplementedError('Not implemented')

# def embed_other_stuff(input_strs):
#     raise NotImplementedError('Not implemented')