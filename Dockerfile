FROM google/cloud-sdk:alpine AS cloud
RUN gsutil -m cp -r gs://text-replace-366410_cloudbuild/text-swap-data.zip .

FROM python:3.9.5 AS base
COPY requirements.txt .
RUN pip install torch==1.10.0+cpu torchvision==0.11.0+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt

FROM python:3.9-slim AS text-swap-api
WORKDIR /app
# ARG PORT
# ENV PORT $PORT
EXPOSE $PORT

COPY --from=base /root/.cache /root/.cache
COPY --from=base requirements.txt .
COPY --from=cloud text-swap-data.zip .

RUN apt-get update
RUN apt-get -y install unzip
RUN unzip text-swap-data.zip
RUN rm text-swap-data.zip

RUN apt-get -y install libgl1
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip install torch==1.10.0+cpu torchvision==0.11.0+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt && rm -rf /root/.cache

COPY . ./

# CMD uvicorn main:app --host 0.0.0.0 --port $PORT
CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --threads 8 main:app
