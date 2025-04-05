import asyncio
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions

# Replace with your actual API key
API_KEY = "qVkgjVw14DWqRVArgPZfCKtH50U8iwdoHw9naZUr0trmCOyi"
default = "../data/Audio_Speech_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"

def get_emotion(result):
    """
    Returns (best_emotion_name, best_emotion_score)
    for the single highest-scoring emotion in the entire result.
    """
    # if not result or not result.prosody or not result.prosody.predictions:
        # return "No emotion data found"
    
    all_emotions = []
    for item in result.prosody.predictions:
        all_emotions.extend(item.emotions)

    best = max(all_emotions, key=lambda e: e.score)
    return best.name

async def analyse(path=default):
    client = AsyncHumeClient(api_key=API_KEY)

    model_config = Config(prosody={})

    stream_options = StreamConnectOptions(config=model_config)

    async with client.expression_measurement.stream.connect(options=stream_options) as socket:
        result = await socket.send_file(path)
    return result
    


