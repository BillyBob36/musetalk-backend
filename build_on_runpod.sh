#!/bin/bash
# Script de build à exécuter sur RunPod

set -e  # Arrêter en cas d'erreur

echo "=========================================="
echo "MuseTalk Docker Build sur RunPod"
echo "=========================================="

# Variables
DOCKER_USERNAME="lamidetlm"
IMAGE_NAME="musetalk-runpod"
IMAGE_TAG="v4"

# 1. Installer Docker si nécessaire
if ! command -v docker &> /dev/null; then
    echo "Installation de Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# 2. Cloner le repository (ou utiliser les fichiers déjà présents)
echo "Préparation des fichiers..."
cd /workspace

# 3. Build l'image Docker
echo "Build de l'image Docker..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# 4. Tag pour Docker Hub
echo "Tag de l'image..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

# 5. Login Docker Hub (les credentials seront passés en variables d'environnement)
echo "Login Docker Hub..."
echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin

# 6. Push vers Docker Hub
echo "Push vers Docker Hub..."
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

echo "=========================================="
echo "✅ Build terminé avec succès !"
echo "Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "=========================================="
