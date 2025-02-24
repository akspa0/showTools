import argparse
import os
import whisper
import librosa
import numpy as np
import json

def detect_dtmf(audio_file):
    """Detects DTMF tones in an audio file.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        list: A list of tuples, where each tuple contains the start and end time
              (in seconds) of a detected DTMF tone sequence.
    """
    y, sr = librosa.load(audio_file)
    # hop_length = 512  # or 256, experiment with this value. Smaller values are more precise but slower.
    hop_length = int(sr * 0.01)  # 10ms hop
    frame_length = int(sr * 0.02) # 20ms frame

    dtmf_tones = []
    onsets = []

    # Use librosa's onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    for onset_time in onset_times:
        start_frame = int(librosa.time_to_frames(onset_time, sr=sr, hop_length=hop_length, n_fft=frame_length))
        end_frame = start_frame + int(0.1 * sr / hop_length)  # Check 0.1 seconds after onset

        if end_frame * hop_length >= len(y):
          end_frame = int(len(y)/hop_length)-1

        # Extract the segment of audio
        y_segment = y[start_frame * hop_length : end_frame * hop_length]

        # Calculate the frequencies present in the segment
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
        magnitudes = np.abs(np.fft.fft(y_segment, n=frame_length))[:len(frequencies)]
        magnitudes_db = librosa.amplitude_to_db(magnitudes, ref=np.max)

        # Define DTMF frequencies
        dtmf_freqs = [697, 770, 852, 941, 1209, 1336, 1477, 1633]
        tolerance = 5  # Frequency tolerance in Hz

        # Check for DTMF tones
        dtmf_detected = False
        for freq in dtmf_freqs:
            lower_bound = freq - tolerance
            upper_bound = freq + tolerance
            peak_indices = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]
            if any(magnitudes_db[peak_indices] > -30):  # Check if magnitude is significant
                dtmf_detected = True
                break

        if dtmf_detected:
            onsets.append(onset_time)

    # Merge close onsets
    if onsets:
      merged_onsets = [onsets[0]]
      for onset in onsets[1:]:
          if onset - merged_onsets[-1] < 0.3:  # Merge if within 300ms
              merged_onsets[-1] = onset
          else:
              dtmf_tones.append((merged_onsets[-1]-0.1, merged_onsets[-1])) #add the onset and 100ms before
              merged_onsets = [onset]
      dtmf_tones.append((merged_onsets[-1]-0.1, merged_onsets[-1]))

    return dtmf_tones

def main():
    parser = argparse.ArgumentParser(description="Audio File Processor")
    parser.add_argument("folder_path", help="Path to the folder containing audio files")
    parser.add_argument("--model", default="turbo", help="Whisper model to use (e.g., tiny, base, small, medium, large)")
    args = parser.parse_args()
    folder_path = args.folder_path
    model_name = args.model

    if not os.path.isdir(folder_path):
        print(f"Error: Folder path '{folder_path}' is not a valid directory.")
        return

    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    print(f"Processing audio files in folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav") or filename.endswith(".mp3") or filename.endswith(".m4a"):
            audio_file_path = os.path.join(folder_path, filename)
            print(f"Transcribing: {audio_file_path}")

            # Detect DTMF tones
            dtmf_times = detect_dtmf(audio_file_path)

            # Transcribe audio file using Whisper API
            result = model.transcribe(audio_file_path, verbose=False, word_timestamps=True)

            # Prepare JSON output
            output_data = {
                "transcription": result["text"],
                "segments": [],
                "dtmf_tones": dtmf_times,
            }

            for segment in result["segments"]:
                segment_data = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "words": [],
                }
                for word in segment["words"]:
                    word_data = {
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                    }
                    segment_data["words"].append(word_data)
                output_data["segments"].append(segment_data)

            # Save JSON output to a file
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(folder_path, output_filename)
            with open(output_path, "w") as outfile:
                json.dump(output_data, outfile, indent=4)

            print(f"Transcription saved to: {output_path}")

if __name__ == "__main__":
    main()