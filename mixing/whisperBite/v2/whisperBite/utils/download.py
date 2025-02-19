"""
Download utilities for WhisperBite.
"""

import os
import logging
from typing import Optional
import yt_dlp
from ..config import AudioProcessingError

logger = logging.getLogger(__name__)

class DownloadUtils:
    @staticmethod
    def download_audio(url: str, output_dir: str) -> str:
        """Download audio from URL."""
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return os.path.join(output_dir, f"{info['title']}.wav")
                
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise AudioProcessingError(f"Audio download failed: {str(e)}")

    @staticmethod
    def get_audio_info(url: str) -> Optional[dict]:
        """Get information about audio without downloading."""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title'),
                    'duration': info.get('duration'),
                    'format': info.get('format'),
                    'url': url
                }
                
        except Exception as e:
            logger.error(f"Failed to get info: {str(e)}")
            return None
