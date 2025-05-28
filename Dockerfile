ARG BASE_IMAGE=python:3.10
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

# Variable d'environnement pour indiquer qu'on est dans Docker
ENV IN_DOCKER=true

WORKDIR /home/kedro_docker

FROM runtime-environment

# copy the whole project with correct ownership
ARG KEDRO_UID=999
ARG KEDRO_GID=0

COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .
# S'assurer que le dossier .kaggle existe avant de copier le fichier kaggle.json
RUN mkdir -p /home/kedro_docker/.kaggle
COPY --chown=${KEDRO_UID}:${KEDRO_GID} kaggle.json /home/kedro_docker/.kaggle/kaggle.json
RUN chmod 600 /home/kedro_docker/.kaggle/kaggle.json

# Créer les dossiers nécessaires APRÈS la copie et avec les bonnes permissions
USER root
RUN mkdir -p data/01_raw \
    data/02_intermediate \
    data/02_model \
    data/03_primary \
    data/04_eval_API \
    data/05_model_output \
    data/05_pred_API \
    data/06_models \
    data/07_predict \
    data/08_outputs \
    data/01_raw/api_images \
    templates \
    model && \
    chown -R ${KEDRO_UID}:${KEDRO_GID} /home/kedro_docker/data && \
    chown -R ${KEDRO_UID}:${KEDRO_GID} /home/kedro_docker/templates && \
    chown -R ${KEDRO_UID}:${KEDRO_GID} /home/kedro_docker/model && \
    chmod -R 755 /home/kedro_docker/data && \
    chmod -R 755 /home/kedro_docker/templates

# Copier le modèle depuis le dossier model vers data si nécessaire
RUN if [ -f model/yolov8n.pt ]; then cp model/yolov8n.pt data/yolov8n.pt; fi && \
    if [ -f model/model.pt ]; then cp model/model.pt data/model.pt; fi && \
    chown -R ${KEDRO_UID}:${KEDRO_GID} /home/kedro_docker/data

# Changer vers l'utilisateur kedro_docker
USER kedro_docker

EXPOSE 5001

RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

CMD ["python", "app.py"]