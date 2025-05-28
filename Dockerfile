ARG BASE_IMAGE=python:3.10-slim
FROM $BASE_IMAGE as runtime-environment

# update pip and install uv
RUN python -m pip install -U "pip>=21.2"
RUN apt-get update && apt-get install -y libgl1
RUN pip install uv

# install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

RUN apt-get update && apt-get install -y tesseract-ocr
ENV TESSERACT_CMD=/usr/bin/tesseract

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

RUN mkdir -p /home/kedro_docker/data/01_raw \
    /home/kedro_docker/data/02_intermediate \
    /home/kedro_docker/data/02_model \
    /home/kedro_docker/data/03_primary \
    /home/kedro_docker/data/04_eval_API \
    /home/kedro_docker/data/05_model_output \
    /home/kedro_docker/data/05_pred_API \
    /home/kedro_docker/data/06_models \
    /home/kedro_docker/data/07_predict \
    /home/kedro_docker/data/08_outputs

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

# copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=999:0 . .

EXPOSE 5001

CMD ["python", "app.py"]