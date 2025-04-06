
import time
import asyncio
from hume import AsyncHumeClient
from hume.expression_measurement.batch import Prosody, Models
from hume.expression_measurement.batch.types import InferenceBaseRequest

API_KEY = "qVkgjVw14DWqRVArgPZfCKtH50U8iwdoHw9naZUr0trmCOyi"

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

    return all_emotions

def extract_emotions(results_list):
    # 1) Take the first InferenceSourcePredictResult
    result = results_list[0]
    
    # 2) Grab the first InferencePrediction
    prediction = results_list[0].results.predictions[0].models.prosody.grouped_predictions[0]
    
    # 3) Access prosody object
    prosody_obj = prediction
    
    if prosody_obj is None or not prosody_obj.grouped_predictions:
        print("No prosody data found.")
        return
    
    # 4) Grab the first grouped_predictions item
    first_group = prosody_obj.grouped_predictions[0]
    
    # 5) The first ProsodyPrediction
    first_pred = first_group.predictions[0]
    
    # 6) The emotions list
    emotions = first_pred.emotions

    return emotions

async def analyse(path):
    # Initialize an authenticated client
    client = AsyncHumeClient(api_key=API_KEY)

    # Define the filepath(s) of the file(s) you would like to analyze
    local_filepaths = [
        open(path, mode="rb")
    ]

    # Create configurations for each model you would like to use (blank = default)
    prosody_config = Prosody()

    # Create a Models object
    models_chosen = Models(prosody=prosody_config)
    
    # Create a stringified object containing the configuration
    stringified_configs = InferenceBaseRequest(models=models_chosen)

    # Start an inference job and print the job_id
    job_id = await client.expression_measurement.batch.start_inference_job_from_local_file(
        json=stringified_configs, file=local_filepaths
    )

    time.sleep(10)

    job_predictions = await client.expression_measurement.batch.get_job_predictions(
        id=job_id
    )
    # print(job_predictions)
    return job_predictions[0]



    


