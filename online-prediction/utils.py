import joblib
import logging
from pydantic import BaseModel
from typing import Optional
from google.cloud import storage
import json
from functools import wraps
from time import time
from google.cloud import bigquery
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report
from sklearn.metrics import average_precision_score, roc_auc_score, plot_precision_recall_curve, plot_roc_curve

logging.basicConfig(level=logging.INFO)

class Input(BaseModel):
    """
    Class with risk model input.
    """

    LIMIT_BAL: int
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: int
    BILL_AMT2: int
    BILL_AMT3: int
    BILL_AMT4: int
    BILL_AMT5: int
    BILL_AMT6: int
    PAY_AMT1: int
    PAY_AMT2: int
    PAY_AMT3: int
    PAY_AMT4: int
    PAY_AMT5: int
    PAY_AMT6: int

class OnlinePredInput(Input):
    """
    Class that extends Input model adding Target for Online preds.
    """
    
    BATCH_FLAG : Optional[int] = 0

class BatchPredInput(BaseModel):
    """
    Batch prediction input.
    """
    BUCKET_NAME: str
    FILEPATH: str

def timeit(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            logging.info(f"Total execution time {func.__name__ }: {end_ if end_ > 0 else 0} ms")
    return _time_it

@timeit
def load_model(path: str):
    """
    load the models from disk
    and put them in a dictionary
    Parameters:
        path (str): File path.

    Returns:
        dict: loaded model
    """
    loaded_model = joblib.load(path)
    logging.info("model loaded from disk")
    return loaded_model

@timeit
def get_json(bucket_name: str, filename: str):
    '''
    this function will get the json object from
    google cloud storage bucket.
    Parameters:
        bucket_name (str): Bucket name in GCP.
        filename (str): filename in GCS.
    Returns:
        json: json data
    '''
    logging.info('Getting file from GCS')
    storage_client = storage.Client()
    BUCKET = storage_client.get_bucket(bucket_name)
    # get the blob
    blob = BUCKET.get_blob(filename)
    # load blob using json
    file_data = json.loads(blob.download_as_string())
    logging.info('Json loaded')
    return file_data

@timeit
def upload_to_bq(df, table_id: str, truncate: str = "WRITE_TRUNCATE"):
    '''
    Uploads dataframe to bq.
    Parameters:
        df (Dataframe): Dataframe to be uploaded.
        table_id (str): Table name in Bigquery.
        truncate (str): Truncate/append.
    Returns:
        None
    '''
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
    write_disposition=truncate,
    )
    job = client.load_table_from_dataframe(
    df, table_id, job_config=job_config
)     
    # Make an API request.
    job.result()  # Wait for the job to complete
    logging.info('Dataframe uploaded to bq')

@timeit
def get_metrics(dataframe):
    '''
    Gets precision - recall - F1 score & support from pred
    Parameters:
        df (Dataframe): Dataframe with pred and true label.
    Returns:
        metrics (dict): Dict with metrics.
    '''
    metrics = classification_report(dataframe['target'], dataframe['prob_1'] > 0.25, output_dict=True)
    return metrics