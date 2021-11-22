# %%
from fastapi import FastAPI
import pandas as pd
from utils import logging
from utils import load_model
from utils import OnlinePredInput
from utils import BatchPredInput
from utils import json
from utils import get_json
from utils import timeit
from utils import upload_to_bq
from utils import get_metrics
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from aiohttp import ClientSession
import asyncio
from fastapi import HTTPException


app = FastAPI()
# %% 
model = load_model('./modelo_default_final.pkl')

@app.get("/healthcheck")
def healthcheck():
    """
    Checks if API is working
    """
    return {"status": "Working"}

@app.post("/risk-online-prediction", description="Risk model: Online Prediction")
async def risk_online_pred(inputs: OnlinePredInput):
    try:
        input_df = pd.DataFrame([jsonable_encoder(inputs)])
        features_df = input_df[[col for col in input_df.columns if col not in ['TARGET','BATCH_FLAG']]]
        result = model.predict_proba(features_df)[0]

        if inputs.BATCH_FLAG == 1:
            return result.tolist()
        else: 
            return JSONResponse(content=json.dumps(result.tolist()))

    except Exception as e:
        logging.error(e)
        raise HTTPException(
            status_code=500, detail="Internal prediction error"
        )


@app.post("/risk-batch-prediction", description="Risk model: Batch Prediction")
async def risk_batch_pred(inputs: BatchPredInput):
    try: 
        json_loaded = get_json(inputs.BUCKET_NAME, inputs.FILEPATH)
        input_df = pd.json_normalize(json_loaded)
        async with ClientSession() as session:
            response = await asyncio.gather(*[risk_online_pred(OnlinePredInput(BATCH_FLAG = 1,**input)) for input in json_loaded])
        df_response = pd.DataFrame(response, columns = ['prob_0', 'prob_1'])
        df_final = pd.merge(input_df, df_response, left_index=True, right_index=True)
        now = pd.to_datetime('now')
        df_final['pred_datetime'] = now
        upload_to_bq(df_final, "tenpo-desafio-data-engineer.riskmodel.risk_model_results", 'WRITE_APPEND')
        try: 
            metrics = get_metrics(df_final)
            df_metrics = pd.DataFrame(metrics).transpose()
            df_metrics['pred_datetime'] = now
            df_metrics.rename(columns={"f1-score": "F1SCORE"}, inplace = True)
            upload_to_bq(df_metrics, "tenpo-desafio-data-engineer.riskmodel.risk_model_metrics", 'WRITE_APPEND')
        except Exception as e:
            logging.info(' Column Target was not detected')
        logging.error(e)
        return('Done')
    except Exception as e:
        logging.error(e)
        raise HTTPException(
            status_code=404, detail="File not found"
        )


    