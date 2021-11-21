gcloud builds submit --project tenpo-desafio-data-engineer --tag gcr.io/tenpo-desafio-data-engineer/risk-model

gcloud run deploy risk-model --image gcr.io/tenpo-desafio-data-engineer/risk-model --platform managed --region us-central1 --project tenpo-desafio-data-engineer --allow-unauthenticated --min-instances 1 --max-instances 1 --memory 2Gi --cpu 4