# [Previous imports remain the same]

class AudioProcessor:
    def process_audio(self, input_path: str, options: ProcessingOptions) -> Dict:
        """Process audio and create archive of results."""
        try:
            results = {}
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            # [Previous processing code remains the same until save_diarization_results]
            
            if options.diarization:
                logger.info("Performing speaker diarization...")
                speaker_segments = self.diarizer.process(
                    normalized_path,
                    features,
                    num_speakers=options.num_speakers
                )
                
                # Save results and get archive path
                diar_results = FileHandler.save_diarization_results(
                    speaker_segments,
                    self.output_dir,
                    input_path,
                    transcription
                )
                
                # Add paths to results
                results['speaker_segments'] = speaker_segments
                results['archive_path'] = diar_results.get('archive_path')
                self.temp_files.extend(diar_results.get('temp_files', []))
                
                logger.info(f"Results archived to: {results['archive_path']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            self.cleanup()
            raise AudioProcessingError(f"Processing pipeline failed: {str(e)}")

    # [Rest of the class methods remain the same]
