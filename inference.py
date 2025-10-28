"""
MuseTalk Inference Pipeline
Génère une vidéo de visage parlant à partir d'un audio et d'une image
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import subprocess
import tempfile
import logging
from typing import Dict, Tuple, Optional
import json
import librosa
import soundfile as sf

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter MuseTalk au path
musetalk_path = '/app/MuseTalk'
logger.info(f"Checking MuseTalk path: {musetalk_path}")
logger.info(f"MuseTalk path exists: {os.path.exists(musetalk_path)}")
if os.path.exists(musetalk_path):
    logger.info(f"MuseTalk contents: {os.listdir(musetalk_path)[:10]}")  # First 10 items
sys.path.insert(0, musetalk_path)
logger.info(f"sys.path: {sys.path[:3]}")  # First 3 paths

try:
    logger.info("Attempting to import MuseTalk modules...")
    from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
    from musetalk.utils.blending import get_image
    from musetalk.utils.utils import load_all_model
    MUSETALK_AVAILABLE = True
    logger.info("✅ MuseTalk modules imported successfully")
except ImportError as e:
    MUSETALK_AVAILABLE = False
    logger.error(f"❌ Failed to import MuseTalk modules: {e}")
    logger.error(f"Import error type: {type(e).__name__}")
    logger.error(f"MuseTalk will not be functional. Check that /app/MuseTalk exists and is properly installed.")
    import traceback
    logger.error(f"Full traceback:\n{traceback.format_exc()}")


class MuseTalkInference:
    """Pipeline d'inférence MuseTalk optimisé pour RunPod"""
    
    def __init__(self, model_path: str = "/app/models/musetalk"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self.models = None
        
        logger.info(f"Initializing MuseTalk on {self.device}")
        
    def load_models(self):
        """Charge les modèles MuseTalk (lazy loading)"""
        if self.models_loaded:
            return
        
        if not MUSETALK_AVAILABLE:
            raise RuntimeError("MuseTalk modules are not available. Cannot load models.")
            
        try:
            logger.info("Loading MuseTalk models...")
            # Charger tous les modèles nécessaires
            self.models = load_all_model()
            self.models_loaded = True
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """
        Prétraite l'audio pour MuseTalk
        
        Args:
            audio_path: Chemin vers le fichier audio
            target_sr: Sample rate cible
            
        Returns:
            Audio array normalisé
        """
        logger.info(f"Preprocessing audio: {audio_path}")
        
        # Charger l'audio avec librosa
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Normaliser
        audio = audio / np.max(np.abs(audio))
        
        logger.info(f"Audio duration: {len(audio)/sr:.2f}s, sample_rate: {sr}")
        return audio
    
    def preprocess_image(self, image_path: str, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Prétraite l'image avatar
        
        Args:
            image_path: Chemin vers l'image
            size: Taille cible (width, height)
            
        Returns:
            Image array normalisée
        """
        logger.info(f"Preprocessing image: {image_path}")
        
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.info(f"Original image shape: {img.shape}, dtype: {img.dtype}")
        
        # Convertir BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionner
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
        
        logger.info(f"Resized image shape: {img_resized.shape}")
        return img_resized
    
    def detect_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        Détecte le visage et extrait la bounding box
        
        Args:
            image: Image array RGB
            
        Returns:
            Dict avec bbox ou None
        """
        logger.info(f"Detecting face... Image shape: {image.shape}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}")
        
        try:
            # Sauvegarder temporairement l'image pour debug
            debug_path = "/tmp/debug_image.jpg"
            cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            logger.info(f"Debug image saved to {debug_path}")
            
            # Utiliser face_alignment pour détecter les landmarks
            import face_alignment
            fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, 
                device=self.device,
                flip_input=False
            )
            
            # Détecter les landmarks (68 points faciaux)
            logger.info("Running face detection with get_landmarks...")
            landmarks = fa.get_landmarks(image)
            
            if landmarks is None or len(landmarks) == 0:
                logger.error("No face detected in image (landmarks is None or empty)")
                return None
            
            logger.info(f"Detected {len(landmarks)} face(s)")
            
            # Prendre le premier visage détecté
            face_landmarks = landmarks[0]
            logger.info(f"Face landmarks shape: {face_landmarks.shape}")
            
            # Calculer la bounding box à partir des landmarks
            x_min, y_min = face_landmarks.min(axis=0).astype(int)
            x_max, y_max = face_landmarks.max(axis=0).astype(int)
            
            bbox_list = [int(x_min), int(y_min), int(x_max), int(y_max)]
            
            logger.info(f"Face detected successfully: bbox={bbox_list}")
            
            return {
                'bbox': bbox_list
            }
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None
    
    def generate_video_frames(
        self, 
        audio: np.ndarray, 
        image: np.ndarray,
        fps: int = 24
    ) -> list:
        """
        Génère les frames vidéo synchronisées avec l'audio
        
        Args:
            audio: Audio array
            image: Image avatar
            fps: Frames per second
            
        Returns:
            Liste de frames (numpy arrays)
        """
        logger.info(f"Generating video frames at {fps} FPS...")
        
        # Charger les modèles si nécessaire
        self.load_models()
        
        # TODO: Implémenter la génération réelle avec MuseTalk
        # Pour l'instant, génération mock pour tester l'infrastructure
        
        duration = len(audio) / 16000  # Assuming 16kHz
        num_frames = int(duration * fps)
        
        logger.info(f"Generating {num_frames} frames for {duration:.2f}s video")
        
        frames = []
        for i in range(num_frames):
            # Mock: copier l'image de base
            # En production: utiliser MuseTalk pour générer les frames animées
            frame = image.copy()
            frames.append(frame)
        
        return frames
    
    def encode_video(
        self,
        frames: list,
        audio_path: str,
        output_path: str,
        fps: int = 24,
        resolution: str = "240p",
        use_nvenc: bool = True
    ) -> str:
        """
        Encode les frames en vidéo MP4 avec audio
        
        Args:
            frames: Liste de frames
            audio_path: Chemin audio source
            output_path: Chemin vidéo sortie
            fps: Frames per second
            resolution: Résolution cible (240p, 360p, 480p)
            use_nvenc: Utiliser NVENC GPU encoding
            
        Returns:
            Chemin de la vidéo générée
        """
        logger.info(f"Encoding video: {resolution} @ {fps}fps")
        
        # Résolutions
        resolutions = {
            "240p": (426, 240),
            "360p": (640, 360),
            "480p": (854, 480)
        }
        
        target_size = resolutions.get(resolution, (426, 240))
        
        # Créer un fichier temporaire pour la vidéo sans audio
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()
        
        # Encoder les frames avec OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            temp_video_path,
            fourcc,
            fps,
            target_size
        )
        
        for frame in frames:
            # Redimensionner si nécessaire
            if frame.shape[:2][::-1] != target_size:
                frame_resized = cv2.resize(frame, target_size)
            else:
                frame_resized = frame
            
            # Convertir RGB -> BGR pour OpenCV
            frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        # Muxer avec l'audio en utilisant FFmpeg
        logger.info("Muxing video with audio...")
        
        codec = "h264_nvenc" if use_nvenc and torch.cuda.is_available() else "libx264"
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-i', temp_video_path,
            '-i', audio_path,
            '-c:v', codec,
            '-preset', 'fast',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-shortest',  # Finir quand le plus court se termine
            output_path
        ]
        
        try:
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Video encoded successfully: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg encoding failed: {e.stderr}")
            raise
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        
        return output_path
    
    def generate(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        fps: int = 24,
        resolution: str = "240p"
    ) -> Dict:
        """
        Pipeline complet de génération
        
        Args:
            audio_path: Chemin fichier audio
            image_path: Chemin image avatar
            output_path: Chemin vidéo sortie
            fps: Frames per second
            resolution: Résolution (240p, 360p, 480p)
            
        Returns:
            Dict avec infos de génération
        """
        logger.info("=" * 50)
        logger.info("Starting MuseTalk generation pipeline")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Image: {image_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Settings: {resolution} @ {fps}fps")
        logger.info("=" * 50)
        
        try:
            # 1. Prétraiter l'audio
            audio = self.preprocess_audio(audio_path)
            
            # 2. Prétraiter l'image
            image = self.preprocess_image(image_path)
            
            # 3. Détecter le visage
            face_info = self.detect_face(image)
            if face_info is None:
                raise ValueError("No face detected in the image")
            
            # 4. Générer les frames vidéo
            frames = self.generate_video_frames(audio, image, fps)
            
            # 5. Encoder la vidéo
            output_video = self.encode_video(
                frames,
                audio_path,
                output_path,
                fps,
                resolution,
                use_nvenc=True
            )
            
            # 6. Récupérer les infos de la vidéo
            video_info = {
                'output_path': output_video,
                'duration': len(audio) / 16000,
                'num_frames': len(frames),
                'fps': fps,
                'resolution': resolution,
                'file_size': os.path.getsize(output_video)
            }
            
            logger.info("Generation completed successfully")
            logger.info(f"Video info: {json.dumps(video_info, indent=2)}")
            
            return video_info
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise


# Test standalone
if __name__ == "__main__":
    inference = MuseTalkInference()
    
    # Test avec des fichiers d'exemple
    result = inference.generate(
        audio_path="/tmp/test_audio.wav",
        image_path="/tmp/test_image.jpg",
        output_path="/tmp/output_video.mp4",
        fps=24,
        resolution="240p"
    )
    
    print(f"Generated video: {result}")
