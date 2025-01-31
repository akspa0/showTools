import re

def extract_quoted_value(pattern, content):
    match = re.search(pattern + r'\s+"([^"]+)"', content)
    return match.group(1) if match else None

def parse_cue(cue_file_path):
    # List of encodings to try
    encodings = ['utf-8', 'cp1252', 'iso-8859-1', 'latin1']
    content = None
    
    for encoding in encodings:
        try:
            with open(cue_file_path, 'r', encoding=encoding) as file:
                content = file.read()
                break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise ValueError(f"Could not decode {cue_file_path} with any of the supported encodings: {encodings}")

    # Extract global metadata
    metadata = {
        'album_artist': extract_quoted_value('PERFORMER', content),
        'album_title': extract_quoted_value('TITLE', content),
        'genre': extract_quoted_value('REM GENRE', content),
        'date': extract_quoted_value('REM DATE', content)
    }

    tracks = []
    track_sections = re.split(r'(?m)^\s*TRACK\s+(\d+)\s+AUDIO', content)[1:]
    
    # Process track sections in pairs (track number and content)
    for i in range(0, len(track_sections), 2):
        track_num = int(track_sections[i])
        track_content = track_sections[i + 1]
        
        track_data = {
            'number': track_num,
            'title': extract_quoted_value('TITLE', track_content),
            'performer': extract_quoted_value('PERFORMER', track_content) or metadata['album_artist']
        }
        
        # Extract INDEX 01 (start) and INDEX 00 (pre-gap) times
        index_01_match = re.search(r'INDEX\s+01\s+(\d+:\d+:\d+)', track_content)
        if index_01_match:
            track_data['start_time'] = index_01_match.group(1).strip()
            # Also store INDEX 00 for gap calculation if available
            index_00_match = re.search(r'INDEX\s+00\s+(\d+:\d+:\d+)', track_content)
            if index_00_match:
                track_data['pregap_time'] = index_00_match.group(1).strip()
            tracks.append(track_data)

    return metadata, tracks