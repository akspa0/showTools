import os
import io
import argparse
import hashlib
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment, silence
from pydub.effects import compress_dynamic_range, normalize
from tqdm import tqdm

def sanitize_filename(filename):
    words = filename.split()[:15]
    filename = ' '.join(words)
    return filename[:128]

def preprocess_audio(audio_segment, normalize_input=True, silence_thresh=-40, compression_ratio=10.0):
    if audio_segment.channels > 1:
        audio_segment = audio_segment.set_channels(1)
    audio_segment = compress_dynamic_range(audio_segment, ratio=compression_ratio)
    if normalize_input:
        audio_segment = normalize(audio_segment)
    return audio_segment

def transcribe_audio_segment(audio_segment, speaker_count, credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    client = speech.SpeechClient()
    with io.BytesIO() as wav_file:
        audio_segment.export(wav_file, format="wav")
        content = wav_file.getvalue()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=audio_segment.frame_rate,
        language_code="en-US",
        enable_speaker_diarization=True,
        diarization_speaker_count=speaker_count,
    )
    print("Sending audio for transcription...")
    response = client.recognize(config=config, audio=audio)
    print("Transcription received.")
    return response

def vad_segment_audio(audio, min_silence_len=1000, silence_thresh=-40, padding_ms=500):
    chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=padding_ms
    )
    print(f"{len(chunks)} segments found with padding added.")
    return chunks

def get_segment_hash(segment_content):
    return hashlib.sha1(segment_content).hexdigest()

def segment_and_transcribe_audio(file_path, output_folder, speaker_count, credentials_path, max_segment_duration, normalize_input, silence_thresh, compression_ratio):
    print(f"Processing directory: {file_path}")
    for filename in os.listdir(file_path):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            file_full_path = os.path.join(file_path, filename)
            print(f"Processing file: {file_full_path}")
            audio = AudioSegment.from_file(file_full_path)
            audio = preprocess_audio(audio, normalize_input, silence_thresh, compression_ratio)

            segments = vad_segment_audio(audio)
            output_folder_for_file = os.path.join(output_folder, os.path.splitext(filename)[0])
            os.makedirs(output_folder_for_file, exist_ok=True)
            flac_output_folder = os.path.join(output_folder_for_file, "flac")
            mp3_output_folder = os.path.join(output_folder_for_file, "mp3")
            os.makedirs(flac_output_folder, exist_ok=True)
            os.makedirs(mp3_output_folder, exist_ok=True)
            duplicates_folder = os.path.join(output_folder_for_file, "duplicates")
            os.makedirs(duplicates_folder, exist_ok=True)
            processed_segments = set()
            segment_transcriptions = {}
            output_txt_path = os.path.join(output_folder_for_file, "transcriptions.txt")
            with open(output_txt_path, "w") as txt_file:
                for i, segment in enumerate(tqdm(segments, desc=f"Processing {filename}", unit="segment")):
                    if len(segment) > max_segment_duration:
                        print(f"Skipping segment {i+1} as it exceeds the maximum duration.")
                        continue
                    segment_hash = get_segment_hash(segment.raw_data)
                    if segment_hash in processed_segments:
                        print(f"Segment {i+1} already processed. Moving to duplicates folder...")
                        segment.export(os.path.join(duplicates_folder, f"segment_{i+1}.flac"), format="flac")
                        continue
                    print(f"Transcribing segment {i+1}/{len(segments)} of file {filename}...")
                    response = transcribe_audio_segment(segment, speaker_count, credentials_path)
                    for j, result in enumerate(response.results):
                        for k, alternative in enumerate(result.alternatives):
                            transcription = alternative.transcript
                            if transcription in segment_transcriptions:
                                print(f"Segment {i+1} transcription already processed. Moving to duplicates folder...")
                                segment.export(os.path.join(duplicates_folder, f"segment_{i+1}.flac"), format="flac")
                            else:
                                segment_transcriptions[transcription] = True
                                unique_filename = sanitize_filename(transcription)
                                # Normalize the output audio segment
                                if normalize_input:
                                    normalized_segment = normalize(segment)
                                else:
                                    normalized_segment = segment
                                normalized_segment.export(os.path.join(flac_output_folder, f"{i+1}_{unique_filename}.flac"), format="flac")
                                normalized_segment.export(os.path.join(mp3_output_folder, f"{i+1}_{unique_filename}.mp3"), format="mp3")
                                txt_file.write(f"Segment {i+1}: {transcription}\n")
                    processed_segments.add(segment_hash)
            print(f"Processing of {filename} complete.")

    # Consolidate all files in "duplicates" folders into a single folder
    consolidate_duplicates(output_folder)

def consolidate_duplicates(output_folder):
    duplicates_folders = []
    for root, dirs, files in os.walk(output_folder):
        if "duplicates" in dirs:
            duplicates_folders.append(os.path.join(root, "duplicates"))
    if duplicates_folders:
        consolidated_duplicates_folder = os.path.join(output_folder, "duplicates")
        os.makedirs(consolidated_duplicates_folder, exist_ok=True)
        for duplicates_folder in duplicates_folders:
            for file in os.listdir(duplicates_folder):
                file_full_path = os.path.join(duplicates_folder, file)
                os.replace(file_full_path, os.path.join(consolidated_duplicates_folder, file))

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio segments and save transcriptions")
    parser.add_argument("--input", type=str, required=True, help="Folder containing audio files to transcribe")
    parser.add_argument("--output", type=str, required=True, help="Folder to save transcriptions and segmented audio")
    parser.add_argument("--speakers", type=int, default=1, help="Number of speakers in the audio (default: 1)")
    parser.add_argument("--credentials", type=str, required=True, help="Path to Google Cloud credentials JSON file")
    parser.add_argument("--max-duration", type=int, default=10000, help="Maximum duration of each segment in milliseconds (default: 10000)")
    parser.add_argument("--normalize-input", action="store_true", help="Normalize input audio before processing (default: False)")
    parser.add_argument("--silence-thresh", type=int, default=-40, help="Silence threshold in dB (default: -40)")
    parser.add_argument("--compression-ratio", type=float, default=10.0, help="Compression ratio for dynamic compression (default: 10.0)")
    args = parser.parse_args()

    segment_and_transcribe_audio(args.input, args.output, args.speakers, args.credentials, args.max_duration, args.normalize_input, args.silence_thresh, args.compression_ratio)

if __name__ == "__main__":
    main()
