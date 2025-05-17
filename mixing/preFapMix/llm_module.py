import logging
from pathlib import Path
import json
import os
import lmstudio as lms
from openai import OpenAI

# Removed all attempts to import submodules for exceptions, as introspection shows they are direct attributes of lms

logger = logging.getLogger(__name__)

def _format_transcript_for_prompt(transcript_data):
    """
    Formats transcript data into a string for the LLM prompt.
    Handles various transcript formats:
    1. JSON list of segments with speaker and text fields
    2. JSON dictionary with segments array
    3. Plain text with speaker annotations
    4. Raw plain text
    """
    logger.debug(f"[LLM Format] Formatting transcript data of type: {type(transcript_data)}")
    
    # Handle already-string data (like plain text transcripts)
    if isinstance(transcript_data, str):
        logger.debug("[LLM Format] Input is already a string, using directly")
        return transcript_data
        
    # Handle list of segment dictionaries (common whisper output format)
    if isinstance(transcript_data, list):
        logger.debug(f"[LLM Format] Input is a list with {len(transcript_data)} items")
        
        # Handle empty list
        if not transcript_data:
            return "Transcript appears to be empty."
            
        formatted_lines = []
        for segment in transcript_data:
            if not isinstance(segment, dict):
                continue  # Skip non-dict items
                
            speaker = segment.get("speaker", "Unknown Speaker")
            text = segment.get("text", "")
            if text:
                formatted_lines.append(f"{speaker}: {text}")
                
        if formatted_lines:
            return "\n".join(formatted_lines)
        else:
            return "Transcript contains no valid text segments."
            
    # Handle dictionary formats
    if isinstance(transcript_data, dict):
        logger.debug(f"[LLM Format] Input is a dictionary with keys: {list(transcript_data.keys())}")
        
        # Check for segments list inside dictionary
        if "segments" in transcript_data and isinstance(transcript_data["segments"], list):
            logger.debug("[LLM Format] Processing 'segments' list from dictionary")
            formatted_lines = []
            for segment in transcript_data["segments"]:
                if not isinstance(segment, dict):
                    continue
                speaker = segment.get("speaker", "Unknown Speaker")
                text = segment.get("text", "")
                if text:
                    formatted_lines.append(f"{speaker}: {text}")
            
            if formatted_lines:
                return "\n".join(formatted_lines)
        
        # Check for direct text field
        if "text" in transcript_data:
            logger.debug("[LLM Format] Using 'text' field from dictionary")
            return transcript_data["text"]
            
        # If transcript key exists
        if "transcript" in transcript_data:
            logger.debug("[LLM Format] Using 'transcript' field from dictionary")
            if isinstance(transcript_data["transcript"], str):
                return transcript_data["transcript"]
            elif isinstance(transcript_data["transcript"], list):
                # Handle array of text chunks
                return " ".join(str(chunk) for chunk in transcript_data["transcript"])
    
    # If we couldn't extract formatted text, return a diagnostic message
    logger.warning(f"[LLM Format] Could not format transcript data properly: {transcript_data[:200] if isinstance(transcript_data, str) else str(transcript_data)[:200]}")
    return "Transcript content could not be properly formatted."

def _format_clap_events_for_prompt(clap_events_data):
    """
    Formats CLAP event data into a string for the LLM prompt.
    Handles cases where events might be missing or malformed.
    Now expects CLAP data to be the parsed JSON from ClapAnnotator v3.8,
    and will extract detections from within the 'stems' structure.
    """
    if not clap_events_data or not isinstance(clap_events_data, dict):
        logger.warning("[LLM Summary] CLAP events data is missing or not in expected dict format.")
        return "No sound events detected or data is malformed."

    all_detections = []
    stems_data = clap_events_data.get("stems")

    if not stems_data or not isinstance(stems_data, dict):
        logger.warning("[LLM Summary] CLAP JSON data does not contain a 'stems' dictionary or it's not a dict.")
        # Fallback: Check if legacy top-level keys exist, though unlikely for this new structure
        legacy_events_list = None
        for key in ["clap_events", "events", "detections", "segments"]:
            if isinstance(clap_events_data.get(key), list):
                legacy_events_list = clap_events_data.get(key)
                logger.warning(f"[LLM Summary] CLAP data missing 'stems', but found legacy list under '{key}'. Processing this list.")
                all_detections = legacy_events_list
                break
        if not legacy_events_list:
            return "Sound event data found but 'stems' structure not recognized or events list is missing."
    else:
        # New structure: Iterate through stems and collect all detections
        for stem_name, stem_details in stems_data.items():
            if isinstance(stem_details, dict):
                detections_list = stem_details.get("detections")
                if isinstance(detections_list, list):
                    for detection in detections_list: # Add stem_name to detection for clarity
                        if isinstance(detection, dict):
                            detection['_stem_source'] = stem_name # Add context about which stem it came from
                            all_detections.append(detection)
                else:
                    logger.warning(f"[LLM Summary] No 'detections' list found for stem '{stem_name}' or it's not a list.")
            else:
                logger.warning(f"[LLM Summary] Stem details for '{stem_name}' is not a dictionary.")

    if not all_detections:
        return "No sound events detected after parsing stems."

    formatted_events = []
    for event in all_detections:
        if not isinstance(event, dict):
            logger.warning(f"[LLM Summary] Skipping non-dict event in collected detections: {event}")
            continue
        label = event.get("label", "Unknown Event")
        start_time = event.get("start_time_seconds", event.get("start_time", "N/A")) # Accommodate different key names potentially
        end_time = event.get("end_time_seconds", event.get("end_time", "N/A"))
        confidence = event.get("confidence", "N/A")
        stem_source = event.get("_stem_source", "main audio") # Default if not added
        
        formatted_events.append(f"- Event: {label} (Source: {stem_source}), Start: {start_time}s, End: {end_time}s, Confidence: {confidence}")
    
    if not formatted_events:
        return "No valid sound events found after parsing."
        
    return "Detected Sound Events:\n" + "\n".join(formatted_events)

def run_llm_tasks(transcript_file_path, config, output_dir=None, output_dir_str=None, **kwargs):
    """
    Flexible LLM task runner. For each task in config['llm_tasks'], renders the prompt_template with transcript/context,
    sends to the LLM, and saves the response to the specified output_file. Supports arbitrary LLM-powered utilities.
    Accepts output_dir or output_dir_str for compatibility with workflow_executor.py.
    Returns a dict of output file paths for workflow context mapping.
    """
    import os
    import logging
    logger = logging.getLogger("llm_module")
    from openai import OpenAI
    
    # Resolve output directory
    if output_dir is None and output_dir_str is not None:
        output_dir = Path(output_dir_str)
    elif output_dir is not None:
        output_dir = Path(output_dir)
    else:
        raise ValueError("run_llm_tasks: Must provide output_dir or output_dir_str.")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[LLM Tasks] Using output directory: {output_dir}")
    
    # Load transcript
    with open(transcript_file_path, 'r', encoding='utf-8') as f:
        transcript = f.read()
    
    llm_tasks = config.get('llm_tasks', [])
    if not llm_tasks:
        logger.warning("No llm_tasks defined in config; nothing to do.")
        return {"llm_outputs": {}}
    
    lm_studio_base_url = config.get('lm_studio_base_url', 'http://localhost:1234/v1')
    lm_studio_api_key = config.get('lm_studio_api_key', 'lm-studio')
    model_id = config.get('lm_studio_model_identifier', 'llama-3.1-8b-supernova-etherealhermes')
    temperature = config.get('lm_studio_temperature', 0.5)
    max_tokens = config.get('lm_studio_max_tokens', 250)
    
    client = OpenAI(base_url=lm_studio_base_url, api_key=lm_studio_api_key)
    
    output_paths = {}
    for task in llm_tasks:
        name = task.get('name', 'unnamed_task')
        prompt_template = task.get('prompt_template', '')
        output_file = task.get('output_file', f'{name}.txt')
        prompt = prompt_template.format(transcript=transcript)
        logger.info(f"[LLM Task: {name}] Sending prompt to model {model_id}...")
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            result = response.choices[0].message.content.strip()
            out_path = os.path.join(output_dir, output_file)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(result)
            logger.info(f"[LLM Task: {name}] Output written to {out_path}")
            output_paths[name] = out_path
        except Exception as e:
            logger.error(f"[LLM Task: {name}] LLM request failed: {e}")
    # For backward compatibility, provide summary_text_file_path as the call_synopsis output if present
    summary_path = output_paths.get('call_synopsis') or next(iter(output_paths.values()), None)
    return {"llm_outputs": output_paths, "summary_text_file_path": summary_path}

# Backward compatibility: alias run_llm_summary to run_llm_tasks with a single-task list if needed

def run_llm_summary(transcript_file_path, config, output_dir, **kwargs):
    """
    Backward-compatible wrapper for old workflow configs. Converts old prompt to a single-task list.
    """
    if 'llm_system_prompt' in config:
        config = config.copy()
        config['llm_tasks'] = [{
            'name': 'llm_summary',
            'prompt_template': config['llm_system_prompt'] + '\nTranscript:\n{transcript}',
            'output_file': 'llm_summary.txt'
        }]
    return run_llm_tasks(transcript_file_path, config, output_dir, **kwargs)

def generate_llm_summary(
    transcript_json_path: Path, 
    system_prompt: str, 
    llm_config: dict, 
    output_dir: Path, 
    output_filename: str
) -> Path | None:
    """
    Generates a summary for a given transcript using a specified system prompt and LLM configuration.
    This function is designed to be called by call_processor.py.

    Args:
        transcript_json_path: Path to the transcript JSON file (e.g., merged_transcript.json).
        system_prompt: The system prompt to guide the LLM.
        llm_config: Dictionary containing LLM configuration, must include 'lm_studio_model_identifier'.
        output_dir: Directory where the summary file will be saved.
        output_filename: Name of the summary file to be created in output_dir.

    Returns:
        Path to the generated summary text file if successful, else None.
    """
    # Set up output path right away
    output_path = Path(output_dir) / output_filename
    logger.info(f"[LLM GenSummary] Output path has been set to: {output_path}")
    
    logger.info(f"[LLM GenSummary] Initiating summary generation for: {transcript_json_path}")
    logger.info(f"[LLM GenSummary] Output target: {output_path}")
    logger.info(f"[LLM GenSummary] System prompt: {system_prompt}")
    logger.info(f"[LLM GenSummary] Config: {llm_config}")

    if not llm_config or 'lm_studio_model_identifier' not in llm_config:
        logger.error("[LLM GenSummary] 'lm_studio_model_identifier' not found in llm_config. Cannot proceed.")
        # Ensure we write an error file even in this case
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Error: LLM model identifier not specified")
            logger.warning(f"[LLM GenSummary] Error file created at: {output_path}")
            return output_path  # Return the path even though it contains an error message
        except Exception as e:
            logger.error(f"[LLM GenSummary] Failed to write error file: {e}")
            return None
    
    lm_studio_model_identifier = llm_config['lm_studio_model_identifier']
    # Record the exact model being used
    logger.info(f"[LLM GenSummary] Using model: {lm_studio_model_identifier}")
    
    # Get base URL and API key, with defaults
    lm_studio_base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
    lm_studio_api_key = llm_config.get('lm_studio_api_key', 'lm-studio')

    # Load and format transcript
    formatted_transcript_text = "Could not load or format transcript."
    if transcript_json_path and transcript_json_path.exists():
        try:
            # Check if this is a plain text file rather than JSON
            if transcript_json_path.suffix.lower() == '.txt':
                logger.info(f"[LLM GenSummary] Detected .txt file, reading as plain text: {transcript_json_path}")
                try:
                    with open(transcript_json_path, 'r', encoding='utf-8') as f_text:
                        formatted_transcript_text = f_text.read()
                    logger.debug(f"[LLM GenSummary] Successfully read plain text file. First 500 chars: {formatted_transcript_text[:500]}")
                except Exception as e_text:
                    logger.error(f"[LLM GenSummary] Error reading plain text file: {e_text}")
                    formatted_transcript_text = f"Error reading plain text transcript: {e_text}"
            else:
                # Debug: log the raw file content first
                with open(transcript_json_path, 'r', encoding='utf-8') as f_debug:
                    raw_content = f_debug.read()
                    logger.debug(f"[LLM GenSummary] Raw transcript file content (first 500 chars): {raw_content[:500]}")
                    
                # Try to parse as JSON
                try:
                    with open(transcript_json_path, 'r', encoding='utf-8') as f_trans:
                        transcript_data = json.load(f_trans)
                    logger.debug(f"[LLM GenSummary] Successfully parsed JSON. Structure type: {type(transcript_data)}")
                    if isinstance(transcript_data, list):
                        logger.debug(f"[LLM GenSummary] JSON is a list with {len(transcript_data)} items")
                        if transcript_data and isinstance(transcript_data[0], dict):
                            logger.debug(f"[LLM GenSummary] First item keys: {list(transcript_data[0].keys())}")
                    elif isinstance(transcript_data, dict):
                        logger.debug(f"[LLM GenSummary] JSON is a dict with keys: {list(transcript_data.keys())}")
                    
                    formatted_transcript_text = _format_transcript_for_prompt(transcript_data)
                    logger.debug(f"[LLM GenSummary] Formatted transcript (first 500 chars): {formatted_transcript_text[:500]}")
                except json.JSONDecodeError as e_json:
                    # If not valid JSON, use the raw content directly
                    logger.warning(f"[LLM GenSummary] Not valid JSON, using raw content directly: {e_json}")
                    formatted_transcript_text = raw_content
                
                # Log a warning if the prompt seems excessively long
                if len(formatted_transcript_text) + len(system_prompt) > 60000: # Conservative threshold
                     logger.warning(f"[LLM GenSummary] Combined length of system prompt and transcript ({len(formatted_transcript_text) + len(system_prompt)} chars) is large. Ensure LM Studio model '{lm_studio_model_identifier}' supports the required context window (e.g., 16384 tokens).")
        except Exception as e:
            logger.error(f"[LLM GenSummary] Error reading/processing transcript: {e}", exc_info=True)
            formatted_transcript_text = f"Error: Could not process transcript file. {e}"
    else:
        logger.warning(f"[LLM GenSummary] Transcript file not found or not provided: {transcript_json_path}")
        # Ensure we write an error file for this case
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Error: Transcript file not found at {transcript_json_path}")
            logger.warning(f"[LLM GenSummary] Error file created at: {output_path}")
            return output_path  # Return the path even though it contains an error message
        except Exception as e:
            logger.error(f"[LLM GenSummary] Failed to write error file: {e}")
            return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_content = f"LLM summary generation failed for model '{lm_studio_model_identifier}'."

    try:
        logger.info(f"[LLM GenSummary] Initializing OpenAI client for LM Studio: {lm_studio_base_url}")
        
        # Use the OpenAI client directly instead of lms.llm, for consistent behavior with run_llm_summary
        try:
            client = OpenAI(
                base_url=lm_studio_base_url,
                api_key=lm_studio_api_key,
            )
            logger.info(f"[LLM GenSummary] OpenAI client initialized for LM Studio.")
        except Exception as client_err:
            logger.error(f"[LLM GenSummary] Failed to initialize OpenAI client: {client_err}", exc_info=True)
            summary_content = f"Failed to initialize OpenAI client: {client_err}"
            # Write the error to file and return its path
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
            return output_path
        
        # The prompt is the transcript content
        user_prompt = formatted_transcript_text
        
        # Configurable parameters for the chat completion
        temperature = llm_config.get('temperature', 0.7)
        max_tokens = llm_config.get('max_response_tokens', 1024)

        logger.info(f"[LLM GenSummary] Sending request to LM Studio model: {lm_studio_model_identifier}. Temp: {temperature}, MaxTokens: {max_tokens}")
        
        # Create the messages array for the API request
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Debug: log the actual API request
        logger.debug(f"[LLM GenSummary] API Request - Messages[0]: {messages[0]}")
        logger.debug(f"[LLM GenSummary] API Request - User prompt (first 200 chars): {user_prompt[:200]}")
        
        # Use OpenAI client's chat.completions.create for better control over system prompt
        try:
            # Extra debug logging
            logger.debug(f"[LLM GenSummary] About to make API call with model: {lm_studio_model_identifier}")
            logger.debug(f"[LLM GenSummary] API parameters: temperature={temperature}, max_tokens={max_tokens}")
            
            completion = client.chat.completions.create(
                model=lm_studio_model_identifier,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Debug: log the full API response
            logger.debug(f"[LLM GenSummary] API Response: {completion}")
            
            if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                summary_content = completion.choices[0].message.content.strip()
                logger.info(f"[LLM GenSummary] Successfully received response from LM Studio model.")
                logger.debug(f"[LLM GenSummary] Response content: {summary_content}")
            else:
                logger.error(f"[LLM GenSummary] Unexpected response structure from LM Studio: {completion}")
                summary_content = "Error: LLM response was empty or malformed."
        
        except Exception as api_err:
            logger.error(f"[LLM GenSummary] API call failed: {api_err}", exc_info=True)
            summary_content = f"Error during API call: {str(api_err)}"
            # Check if the error is a connection error
            if "Connection refused" in str(api_err):
                logger.error("[LLM GenSummary] Connection error - Is LM Studio running and accessible?")
                summary_content = "Error: Could not connect to LM Studio. Please ensure it is running."

    except Exception as e_other:
        summary_content = f"Error during LLM processing: {str(e_other)}"
        logger.error(f"[LLM GenSummary] Unexpected error: {e_other}", exc_info=True)

    # Always write a file, even if it's an error message
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        logger.info(f"[LLM GenSummary] Output saved to: {output_path}")
        
        # Return the path regardless of success/failure
        logger.info(f"[LLM GenSummary] Returning output path: {output_path}")
        return output_path
        
    except Exception as e_write:
        logger.error(f"[LLM GenSummary] Failed to write output file: {e_write}", exc_info=True)
        return None 