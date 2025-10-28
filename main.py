"""
RunPod Serverless Handler pour MuseTalk
Endpoint: /generate
Input: {"audio_url": "...", "image_url": "...", "fps": 24, "resolution": "240p"}
Output: {"video_url": "...", "duration": 10.5, "num_frames": 252}
"""

import runpod
import os
import sys
import tempfile
import logging
import time
import base64
from pathlib import Path
from typing import Dict, Any
import requests
from inference import MuseTalkInference

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialiser le pipeline d'inférence (global pour réutilisation)
inference_pipeline = None


def download_file(url: str, output_path: str) -> str:
    """
    Télécharge un fichier depuis une URL ou décode un data URL
    
    Args:
        url: URL du fichier ou data URL (base64)
        output_path: Chemin de destination
        
    Returns:
        Chemin du fichier téléchargé
    """
    logger.info(f"Processing file from URL (length: {len(url)} chars)")
    
    try:
        # Vérifier si c'est un data URL (base64)
        if url.startswith('data:'):
            logger.info("Detected data URL, decoding base64...")
            # Format: data:audio/wav;base64,XXXXX ou data:image/jpeg;base64,XXXXX
            header, encoded = url.split(',', 1)
            file_bytes = base64.b64decode(encoded)
            
            with open(output_path, 'wb') as f:
                f.write(file_bytes)
            
            logger.info(f"File decoded: {output_path} ({os.path.getsize(output_path)} bytes)")
            return output_path
        else:
            # URL HTTP classique
            logger.info(f"Downloading from HTTP URL: {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"File downloaded: {output_path} ({os.path.getsize(output_path)} bytes)")
            return output_path
        
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        raise


def encode_video_to_base64(video_path: str) -> str:
    """
    Encode une vidéo en base64 pour le retour
    
    Args:
        video_path: Chemin de la vidéo
        
    Returns:
        String base64
    """
    with open(video_path, 'rb') as f:
        video_bytes = f.read()
        return base64.b64encode(video_bytes).decode('utf-8')


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler principal RunPod Serverless
    
    Args:
        job: Dict avec job_input contenant:
            - audio_url: URL du fichier audio
            - image_url: URL de l'image avatar
            - fps: Frames per second (15, 24, 30)
            - resolution: Résolution (240p, 360p, 480p)
            
    Returns:
        Dict avec résultat ou erreur
    """
    global inference_pipeline
    
    job_input = job.get('input', {})
    
    # Validation des inputs
    audio_url = job_input.get('audio_url')
    image_url = job_input.get('image_url')
    fps = job_input.get('fps', 24)
    resolution = job_input.get('resolution', '240p')
    
    if not audio_url or not image_url:
        return {
            'error': 'Missing required parameters: audio_url and image_url'
        }
    
    # Valider les paramètres
    valid_fps = [15, 24, 30]
    valid_resolutions = ['240p', '360p', '480p']
    
    if fps not in valid_fps:
        return {'error': f'Invalid fps. Must be one of {valid_fps}'}
    
    if resolution not in valid_resolutions:
        return {'error': f'Invalid resolution. Must be one of {valid_resolutions}'}
    
    logger.info("=" * 60)
    logger.info(f"New job received:")
    logger.info(f"  Audio URL: {audio_url}")
    logger.info(f"  Image URL: {image_url}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Resolution: {resolution}")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Initialiser le pipeline si nécessaire
        if inference_pipeline is None:
            logger.info("Initializing inference pipeline...")
            
            # Vérifier l'état de MuseTalk avant d'initialiser
            from inference import MUSETALK_AVAILABLE
            logger.info(f"MuseTalk modules available: {MUSETALK_AVAILABLE}")
            
            if not MUSETALK_AVAILABLE:
                logger.error("Cannot initialize pipeline: MuseTalk modules are not available")
                logger.error("Please check the container logs at startup for import errors")
                return {
                    'error': 'MuseTalk modules failed to import. Check container startup logs for details.'
                }
            
            inference_pipeline = MuseTalkInference()
        
        # Créer des fichiers temporaires
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Télécharger l'audio
            audio_ext = Path(audio_url).suffix or '.wav'
            audio_path = temp_dir_path / f"input_audio{audio_ext}"
            download_file(audio_url, str(audio_path))
            
            # Télécharger l'image
            image_ext = Path(image_url).suffix or '.jpg'
            image_path = temp_dir_path / f"input_image{image_ext}"
            download_file(image_url, str(image_path))
            
            # Chemin de sortie
            output_path = temp_dir_path / "output_video.mp4"
            
            # Générer la vidéo
            logger.info("Starting video generation...")
            generation_start = time.time()
            
            result = inference_pipeline.generate(
                audio_path=str(audio_path),
                image_path=str(image_path),
                output_path=str(output_path),
                fps=fps,
                resolution=resolution
            )
            
            generation_time = time.time() - generation_start
            logger.info(f"Video generation completed in {generation_time:.2f}s")
            
            # Encoder la vidéo en base64 pour le retour
            logger.info("Encoding video to base64...")
            video_base64 = encode_video_to_base64(str(output_path))
            
            total_time = time.time() - start_time
            
            # Préparer la réponse
            response = {
                'video_base64': video_base64,
                'duration': result['duration'],
                'num_frames': result['num_frames'],
                'fps': fps,
                'resolution': resolution,
                'file_size': result['file_size'],
                'processing_time': round(total_time, 2),
                'generation_time': round(generation_time, 2)
            }
            
            logger.info("=" * 60)
            logger.info(f"Job completed successfully in {total_time:.2f}s")
            logger.info(f"  Video duration: {result['duration']:.2f}s")
            logger.info(f"  Frames: {result['num_frames']}")
            logger.info(f"  File size: {result['file_size']} bytes")
            logger.info("=" * 60)
            
            return response
            
    except Exception as e:
        error_msg = f"Job failed: {str(e)}"
        logger.error(error_msg)
        logger.exception(e)
        
        return {
            'error': error_msg,
            'processing_time': round(time.time() - start_time, 2)
        }


# Point d'entrée RunPod Serverless
if __name__ == "__main__":
    logger.info("Starting RunPod Serverless Worker for MuseTalk")
    logger.info(f"CUDA Available: {os.system('nvidia-smi') == 0}")
    
    # Démarrer le worker RunPod
    runpod.serverless.start({
        "handler": handler
    })
