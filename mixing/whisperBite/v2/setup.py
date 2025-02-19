"""
Setup configuration for WhisperBite package.
"""

from setuptools import setup, find_packages

setup(
    name="whisperBite",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "whisper>=1.0.0",
        "pyannote.audio>=3.0.0",
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pydub>=0.25.1",
        "yt-dlp>=2023.3.4",
        "ffmpeg-python>=0.2.0",
        "demucs>=4.0.0"
    ],
    entry_points={
        "console_scripts": [
            "whisperbite=whisperBite.main:main",
        ],
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced audio processing and transcription tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/whisperBite",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
