# MuseTalk RunPod Serverless - Optimized for RTX 4090
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Éviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Créer le répertoire de travail
WORKDIR /app

# Cloner MuseTalk repository en premier
RUN git clone https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk

# Installer les dépendances de MuseTalk depuis leur requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /app/MuseTalk/requirements.txt

# Copier et installer nos requirements supplémentaires (RunPod, etc.)
COPY requirements.txt /app/requirements_extra.txt
RUN pip3 install --no-cache-dir -r /app/requirements_extra.txt

# Créer les dossiers pour les modèles (seront téléchargés au premier lancement)
RUN mkdir -p /app/models/musetalk && \
    mkdir -p /app/models/dwpose && \
    mkdir -p /app/models/face-parse-bisent && \
    mkdir -p /app/models/sd-vae-ft-mse && \
    mkdir -p /root/.cache/torch/hub/checkpoints

# Télécharger les modèles nécessaires
RUN python3 -c "import face_alignment; import torch; fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False); print('Face alignment models loaded successfully')" || true

# Copier les scripts d'inférence
COPY inference.py /app/
COPY main.py /app/
COPY check_musetalk.py /app/

# Vérifier l'installation de MuseTalk
RUN python3 /app/check_musetalk.py || echo 'Check failed but continuing...'

# Créer les dossiers temporaires
RUN mkdir -p /tmp/musetalk_input /tmp/musetalk_output

# Exposer le port pour RunPod
EXPOSE 8000

# Point d'entrée RunPod Serverless
CMD ["python3", "-u", "main.py"]
