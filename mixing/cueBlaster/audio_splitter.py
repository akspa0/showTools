import os
import re
from pydub import AudioSegment
from mutagen.flac import FLAC
from mutagen.id3 import ID3, TIT2, TPE1, TALB, TRCK, TDRC, TCON

def sanitize_filename(filename):
    # Remove invalid characters for Windows filenames
    # Replace with underscore: \ / : * ? " < > |
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', filename)
    # Remove any leading/trailing spaces or dots
    sanitized = sanitized.strip('. ')
    return sanitized

def apply_metadata(output_file, track_data, metadata):
    if output_file.lower().endswith('.flac'):
        audio = FLAC(output_file)
        audio['title'] = track_data['title']
        audio['artist'] = track_data['performer']
        audio['album'] = metadata['album_title']
        audio['tracknumber'] = str(track_data['number'])
        if metadata.get('date'):
            audio['date'] = metadata['date']
        if metadata.get('genre'):
            audio['genre'] = metadata['genre']
        audio.save()
    elif output_file.lower().endswith('.mp3'):
        audio = ID3(output_file)
        audio.add(TIT2(encoding=3, text=track_data['title']))
        audio.add(TPE1(encoding=3, text=track_data['performer']))
        audio.add(TALB(encoding=3, text=metadata['album_title']))
        audio.add(TRCK(encoding=3, text=str(track_data['number'])))
        if metadata.get('date'):
            audio.add(TDRC(encoding=3, text=metadata['date']))
        if metadata.get('genre'):
            audio.add(TCON(encoding=3, text=metadata['genre']))
        audio.save()

def time_to_ms(time_str):
    """Convert CUE sheet time format (MM:SS:FF) to milliseconds"""
    minutes, seconds, frames = map(int, time_str.split(':'))
    return ((minutes * 60 + seconds) * 1000) + (frames * 1000 // 75)

def split_audio(cue_data, audio_file_path, output_dir=None):
    """Split audio file according to CUE sheet data.
    
    Args:
        cue_data: Tuple of (metadata, tracks) from parse_cue
        audio_file_path: Path to the source audio file
        output_dir: Custom output directory (optional)
    """
    metadata, tracks = cue_data
    audio = AudioSegment.from_file(audio_file_path)

    # If no output directory specified, create 'split' subdirectory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(audio_file_path), 'split')
    
    os.makedirs(output_dir, exist_ok=True)

    for i, track in enumerate(tracks):
        # Convert time to milliseconds using INDEX 01 (actual start)
        start_ms = time_to_ms(track['start_time'])
        
        # Calculate end time from next track's INDEX 00 (pre-gap) or end of audio
        if i == len(tracks) - 1:
            end_ms = len(audio)  # Use actual audio length instead of duration_seconds
        else:
            # Use next track's INDEX 00 (pregap) as the end point if available
            next_track = tracks[i + 1]
            if 'pregap_time' in next_track:
                end_ms = time_to_ms(next_track['pregap_time'])
            else:
                # If no pregap, use start time
                end_ms = time_to_ms(next_track['start_time'])
        
        track_segment = audio[start_ms:end_ms]
        
        # Create filename with track number and title
        safe_title = sanitize_filename(track['title'])
        track_number = str(track['number']).zfill(2)  # Pad with leading zero
        output_file = os.path.join(output_dir, f'{track_number} - {safe_title}.{audio_file_path.split(".")[-1]}')
        
        # Export the track
        track_segment.export(output_file, format=audio_file_path.split(".")[-1])
        
        # Apply metadata
        apply_metadata(output_file, track, metadata)
        
        print(f'Split track {track_number}: {output_file}')