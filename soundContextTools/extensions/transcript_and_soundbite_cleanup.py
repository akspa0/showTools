import shutil
import re
from pathlib import Path
from extension_base import ExtensionBase

class TranscriptAndSoundbiteCleanup(ExtensionBase):
    def run(self):
        self.log("Starting transcript aggregation and soundbites cleanup.")
        self.aggregate_transcripts()
        self.cleanup_soundbites()
        self.log("Extension completed.")

    def aggregate_transcripts(self):
        finalized = self.output_root / 'finalized'
        calls_dir = finalized / 'calls'
        speakers_dir = finalized / 'speakers'
        transcripts_dir = finalized / 'transcripts'
        
        # Aggregate call-level transcripts
        if calls_dir.exists() and transcripts_dir.exists():
            for call_folder in calls_dir.iterdir():
                if call_folder.is_dir():
                    call_id = call_folder.name
                    transcript_file = transcripts_dir / f"{call_id}.txt"
                    if transcript_file.exists():
                        dest = call_folder / 'call_transcript.txt'
                        shutil.copy2(transcript_file, dest)
                        self.log(f"Copied transcript for {call_id} to {dest}")

        # Aggregate speaker transcripts into call folders if possible
        if speakers_dir.exists() and calls_dir.exists():
            for speaker_folder in speakers_dir.iterdir():
                if speaker_folder.is_dir():
                    for seg_file in speaker_folder.glob('*.txt'):
                        # Try to infer call association from filename (e.g., 0000-...txt)
                        match = re.match(r'(\d{4})-', seg_file.name)
                        if match:
                            call_id = match.group(1)
                            call_folder = calls_dir / call_id
                            if call_folder.exists():
                                dest = call_folder / f"speaker_{speaker_folder.name}_{seg_file.name}"
                                shutil.copy2(seg_file, dest)
                                self.log(f"Copied speaker transcript {seg_file.name} to {dest}")

    def cleanup_soundbites(self):
        soundbites_dir = self.output_root / 'finalized' / 'soundbites'
        if not soundbites_dir.exists():
            self.log("No soundbites directory found.")
            return
        for folder in soundbites_dir.iterdir():
            if folder.is_dir() and re.fullmatch(r'\d{4}', folder.name):
                # Check if folder is obsolete (not referenced in manifest or not matching new naming)
                shutil.rmtree(folder)
                self.log(f"Removed obsolete soundbites folder: {folder}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcript_and_soundbite_cleanup.py <output_root>")
        exit(1)
    ext = TranscriptAndSoundbiteCleanup(sys.argv[1])
    ext.run() 