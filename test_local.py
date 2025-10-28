"""
Script de test local pour le pipeline MuseTalk (sans RunPod)
Permet de tester l'inférence localement avant déploiement
"""

import sys
import os
from pathlib import Path
import logging

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importer le pipeline
try:
    from inference import MuseTalkInference
except ImportError as e:
    logger.error(f"Failed to import inference module: {e}")
    logger.info("Make sure you have installed all dependencies from requirements.txt")
    sys.exit(1)


def test_inference():
    """Test du pipeline d'inférence"""
    
    logger.info("=" * 60)
    logger.info("MuseTalk Local Test")
    logger.info("=" * 60)
    
    # Créer le dossier de test
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Vérifier les fichiers de test
    audio_path = test_dir / "test_audio.wav"
    image_path = test_dir / "test_image.jpg"
    output_path = test_dir / "output_video.mp4"
    
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        logger.info("Please add a test audio file (WAV format) to test_data/test_audio.wav")
        return False
    
    if not image_path.exists():
        logger.warning(f"Image file not found: {image_path}")
        logger.info("Please add a test image file (JPG format) to test_data/test_image.jpg")
        return False
    
    # Initialiser le pipeline
    logger.info("Initializing MuseTalk pipeline...")
    inference = MuseTalkInference()
    
    # Tester la génération
    try:
        logger.info("Starting video generation...")
        result = inference.generate(
            audio_path=str(audio_path),
            image_path=str(image_path),
            output_path=str(output_path),
            fps=24,
            resolution="240p"
        )
        
        logger.info("=" * 60)
        logger.info("✅ Test completed successfully!")
        logger.info(f"Output video: {result['output_path']}")
        logger.info(f"Duration: {result['duration']:.2f}s")
        logger.info(f"Frames: {result['num_frames']}")
        logger.info(f"File size: {result['file_size']} bytes")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.exception(e)
        return False


def test_mock_handler():
    """Test du handler RunPod en mode mock"""
    
    logger.info("=" * 60)
    logger.info("Testing RunPod Handler (Mock Mode)")
    logger.info("=" * 60)
    
    # Simuler un job RunPod
    mock_job = {
        "input": {
            "audio_url": "file://test_data/test_audio.wav",
            "image_url": "file://test_data/test_image.jpg",
            "fps": 24,
            "resolution": "240p"
        }
    }
    
    try:
        from main import handler
        
        logger.info("Calling handler with mock job...")
        result = handler(mock_job)
        
        if "error" in result:
            logger.error(f"❌ Handler returned error: {result['error']}")
            return False
        
        logger.info("=" * 60)
        logger.info("✅ Handler test completed successfully!")
        logger.info(f"Processing time: {result.get('processing_time')}s")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Handler test failed: {e}")
        logger.exception(e)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MuseTalk pipeline locally")
    parser.add_argument(
        "--mode",
        choices=["inference", "handler", "all"],
        default="all",
        help="Test mode"
    )
    
    args = parser.parse_args()
    
    success = True
    
    if args.mode in ["inference", "all"]:
        success = test_inference() and success
    
    if args.mode in ["handler", "all"]:
        success = test_mock_handler() and success
    
    if success:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed")
        sys.exit(1)
