FROM --platform=linux/amd64 python:3.10

# Prévenir les erreurs matplotlib et Ultralytics en environnement restreint
ENV MPLCONFIGDIR=/tmp
ENV YOLO_CONFIG_DIR=/tmp

# update pip and install uv
RUN python -m pip install -U "pip>=21.2"

# Installer les dépendances système nécessaires pour OpenCV et OCR
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgdal-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Télécharger les modèles EasyOCR à l'avance
RUN python -c "import easyocr; easyocr.Reader(['en'], download_enabled=True)"

# install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache-dir -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# Définir les variables d'environnement pour optimiser les performances
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1

# Variables pour désactiver les avertissements
ENV PYTHONWARNINGS="ignore"
ENV TF_CPP_MIN_LOG_LEVEL=2

# DÉSACTIVER KAGGLE COMPLÈTEMENT
ENV KAGGLE_CONFIG_DIR=/tmp
ENV KAGGLE_USERNAME=disabled
ENV KAGGLE_KEY=disabled

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

# Variable d'environnement pour indiquer qu'on est dans Docker
ENV IN_DOCKER=true

WORKDIR /home/kedro_docker

# copy the whole project with correct ownership
ARG KEDRO_UID=999
ARG KEDRO_GID=0

COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

# Copier le fichier kaggle.json si fourni
COPY kaggle.json /home/kedro_docker/.config/kaggle/kaggle.json
RUN chmod 600 /home/kedro_docker/.config/kaggle/kaggle.json

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

# Définir la variable d'environnement KEDRO_PROJECT_PATH
ENV KEDRO_PROJECT_PATH=/home/kedro_docker

EXPOSE 5001

# Vérification du contenu du répertoire
RUN echo "📁 Structure des fichiers dans /home/kedro_docker :" && ls -la /home/kedro_docker

CMD ["python", "app.py"]