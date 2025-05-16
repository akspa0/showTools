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
    Handles both structured segments and plain text.
    """
    if isinstance(transcript_data, dict):
        if "segments" in transcript_data and isinstance(transcript_data["segments"], list):
            formatted_lines = []
            for segment in transcript_data["segments"]:
                speaker = segment.get("speaker", "Unknown Speaker")
                text = segment.get("text", "")
                formatted_lines.append(f"{speaker}: {text}")
            return "\\n".join(formatted_lines)
        elif "text" in transcript_data:
            return transcript_data["text"]
    elif isinstance(transcript_data, str): # For very simple plain text transcripts
        return transcript_data
    return "Transcript content not available or in an unrecognized format."

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

def run_llm_summary(
    transcript_file_path: str, 
    clap_events_file_path: str, 
    diarization_file_path: str, # Kept for signature consistency, may be used later
    output_dir_str: str, 
    pii_safe_file_prefix: str = None, # Added to match executor call, marked as unused for now
    config: dict = None
):
    """
    Performs LLM-based summarization and analysis using a local LM Studio instance.
    pii_safe_file_prefix is passed from the workflow executor but not currently used by this function.
    """
    if config is None:
        config = {}

    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[LLM Summary] Running LLM summary/analysis via LM Studio.")
    logger.info(f"[LLM Summary] Transcript: {transcript_file_path}")
    logger.info(f"[LLM Summary] CLAP Events: {clap_events_file_path}")
    logger.info(f"[LLM Summary] Diarization: {diarization_file_path}")
    logger.info(f"[LLM Summary] Config: {config}")
    logger.info(f"[LLM Summary] Output directory: {output_dir}")

    # LM Studio Model Identifier from config
    lm_studio_model_identifier = config.get('lm_studio_model_identifier', 'NousResearch/Hermes-2-Pro-Llama-3-8B') 
    # LM Studio Base URL from config
    lm_studio_base_url = config.get('lm_studio_base_url', 'http://localhost:1234/v1')
    lm_studio_api_key = config.get('lm_studio_api_key', 'lm-studio') # Often 'lm-studio' or can be empty

    logger.info(f"[LLM Summary] Attempting to use LM Studio model: {lm_studio_model_identifier} via {lm_studio_base_url}")

    # Load and format transcript
    transcript_prompt_text = "Could not load transcript."
    if transcript_file_path and Path(transcript_file_path).exists():
        try:
            with open(transcript_file_path, 'r', encoding='utf-8') as f_trans:
                # Directly read the content of the .txt file
                transcript_prompt_text = f_trans.read()
                if not transcript_prompt_text.strip(): # Check if file is empty or only whitespace
                    transcript_prompt_text = "Transcript file was found but is empty."
                    logger.warning(f"[LLM Summary] Transcript file {transcript_file_path} is empty.")
                else:
                    logger.info(f"[LLM Summary] Successfully loaded transcript text from {transcript_file_path}.")
        except Exception as e_read_trans:
            transcript_prompt_text = f"Error reading transcript file: {e_read_trans}"
            logger.error(f"[LLM Summary] Error reading transcript file {transcript_file_path}: {e_read_trans}")
    else:
        logger.warning(f"[LLM Summary] Transcript file not found or not provided: {transcript_file_path}")
        transcript_prompt_text = "Transcript file path was not provided or the file does not exist." # More specific message

    # Load and format CLAP events
    clap_events_prompt_text = "No CLAP events information available."
    if clap_events_file_path and Path(clap_events_file_path).exists():
        try:
            with open(clap_events_file_path, 'r', encoding='utf-8') as f_clap:
                clap_data = json.load(f_clap)
                clap_events_prompt_text = _format_clap_events_for_prompt(clap_data)
        except Exception as e_read_clap:
            clap_events_prompt_text = f"Error reading or parsing CLAP events: {e_read_clap}"
            logger.error(f"[LLM Summary] Error reading CLAP events file {clap_events_file_path}: {e_read_clap}")
    else:
        logger.warning(f"[LLM Summary] CLAP events file not found or not provided: {clap_events_file_path}")

    # Construct Prompt
    prompt = f"""
You are an AI assistant. Your task is to provide a concise summary of the following conversation transcript.
Incorporate information about notable sound events if they provide relevant context.

Conversation Transcript:
---
{transcript_prompt_text}
---

Notable Sound Events Information:
---
{clap_events_prompt_text}
---

Please generate a summary of the conversation:
"""
    
    summary_content = "LLM Summarization via LM Studio failed."
    final_summary_file_path = output_dir / "final_analysis_summary.txt"

    try:
        logger.info(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] Initializing OpenAI client for LM Studio: {lm_studio_base_url}")
        
        client = OpenAI(
            base_url=lm_studio_base_url,
            api_key=lm_studio_api_key, # api_key is often not strictly required by LM Studio but good to include
        )
        
        logger.info(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] OpenAI client initialized for LM Studio.")
        
        # The prompt is already constructed above
        # Configurable parameters for the chat completion, e.g., temperature, max_tokens
        temperature = config.get('lm_studio_temperature', 0.7)
        max_tokens = config.get('lm_studio_max_tokens', 500)
        # System prompt can also be made configurable
        system_prompt_content = config.get('llm_system_prompt', "You are an AI assistant. Your task is to provide a concise summary of the conversation transcript. Incorporate information about notable sound events if they provide relevant context.")

        logger.info(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] Sending request to LM Studio model: {lm_studio_model_identifier}. Temp: {temperature}, MaxTokens: {max_tokens}")
        
        completion = client.chat.completions.create(
            model=lm_studio_model_identifier,
            messages=[
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": prompt} # 'prompt' contains the main user-facing prompt and data
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            summary_content = completion.choices[0].message.content.strip()
            logger.info(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] Summary received successfully from LM Studio.")
        else:
            logger.error(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] Unexpected response structure from LM Studio: {completion}")
            summary_content = "LLM Summarization failed due to unexpected response structure."

    except lms.LMStudioAPIError as e: # Catching specific lmstudio errors if they still can occur through OpenAI client or underlying http
        logger.error(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] LM Studio API Error: {e}")
        summary_content = f"LLM Summarization failed due to LM Studio API Error: {e}"
    except Exception as e: # General catch-all for other errors like httpx.ConnectError
        logger.error(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] LM Studio Client Error:\\n    {e}")
        summary_content = f"LLM Summarization failed due to Client Error: {e}"
        # More detailed error logging for connection issues
        if "All connection attempts failed" in str(e) or "Connection refused" in str(e):
            logger.error(f"    Is LM Studio running and accessible at {lm_studio_base_url}?.")

    try:
        with open(final_summary_file_path, 'w', encoding='utf-8') as f_sum:
            f_sum.write(summary_content)
        logger.info(f"[LLM Summary] Summary saved to: {final_summary_file_path}")
        if "failed" in summary_content.lower() or "error" in summary_content.lower() or "ensure lm studio is running" in summary_content.lower():
             return {"summary_text_file_path": str(final_summary_file_path), "analysis_complete": False, "error": summary_content}
        return {"summary_text_file_path": str(final_summary_file_path), "analysis_complete": True}
        
    except Exception as e_write:
        logger.error(f"[LLM Summary] Failed to write summary file: {e_write}")
        return {"error": f"Failed to write summary file: {e_write}", "summary_text_file_path": None, "analysis_complete": False} 

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
    logger.info(f"[LLM GenSummary] Initiating summary generation for: {transcript_json_path}")
    logger.info(f"[LLM GenSummary] Output target: {output_dir / output_filename}")

    if not llm_config or 'lm_studio_model_identifier' not in llm_config:
        logger.error("[LLM GenSummary] 'lm_studio_model_identifier' not found in llm_config. Cannot proceed.")
        return None
    
    lm_studio_model_identifier = llm_config['lm_studio_model_identifier']

    # Load and format transcript
    formatted_transcript_text = "Could not load or format transcript."
    if transcript_json_path and transcript_json_path.exists():
        try:
            with open(transcript_json_path, 'r', encoding='utf-8') as f_trans:
                transcript_data = json.load(f_trans)
            formatted_transcript_text = _format_transcript_for_prompt(transcript_data)
            # Log a warning if the prompt seems excessively long, as a hint for context window issues.
            # This is a rough check; actual token count depends on the model's tokenizer.
            # Assuming average 4 chars per token, 16384 tokens ~ 65536 chars
            # System prompt length also contributes.
            if len(formatted_transcript_text) + len(system_prompt) > 60000: # Conservative threshold
                 logger.warning(f"[LLM GenSummary] Combined length of system prompt and transcript ({len(formatted_transcript_text) + len(system_prompt)} chars) is large. Ensure LM Studio model '{lm_studio_model_identifier}' supports the required context window (e.g., 16384 tokens).")
        except json.JSONDecodeError as e_json:
            logger.error(f"[LLM GenSummary] Error decoding JSON from transcript file {transcript_json_path}: {e_json}")
            formatted_transcript_text = f"Error: Could not decode transcript JSON. {e_json}"
        except IOError as e_io:
            logger.error(f"[LLM GenSummary] File I/O error reading transcript file {transcript_json_path}: {e_io}")
            formatted_transcript_text = f"Error: Could not read transcript file. {e_io}"
        except Exception as e_fmt:
            logger.error(f"[LLM GenSummary] Unexpected error formatting transcript from {transcript_json_path}: {e_fmt}")
            formatted_transcript_text = f"Error: Unexpected error processing transcript. {e_fmt}"
    else:
        logger.warning(f"[LLM GenSummary] Transcript file not found or not provided: {transcript_json_path}")
        # Return None because we need a transcript for this function
        return None 

    output_path = Path(output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_content = f"LLM summary generation failed for model '{lm_studio_model_identifier}'."

    try:
        logger.info(f"[LLM GenSummary] Initializing LM Studio model handle: {lm_studio_model_identifier}")
        # --- Use Convenience API as per docs --- 
        model = lms.llm(lm_studio_model_identifier) # Get model handle
        logger.info(f"[LLM GenSummary] Model handle obtained for {lm_studio_model_identifier}")
        
        # The prompt is already constructed above
        # Configurable parameters for the chat completion, e.g., temperature, max_tokens
        temperature = llm_config.get("temperature", 0.7)
        max_tokens = llm_config.get("max_response_tokens", 1024)

        logger.info(f"[LLM GenSummary] Sending request to LM Studio model: {lm_studio_model_identifier}. Temp: {temperature}, MaxTokens: {max_tokens}")
        
        # Use model.respond() for consistency
        response = model.respond(formatted_transcript_text, config=completion_params)

        if isinstance(response, str):
            summary_content = response.strip()
        elif hasattr(response, 'content') and isinstance(response.content, str):
            summary_content = response.content.strip()
        # Add more robust response parsing if needed, similar to run_llm_summary
        else:
            logger.error(f"[LLM GenSummary] Unexpected response structure from model.respond(): {type(response)} - {str(response)[:500]}")
            summary_content = "Error: LLM response was empty or malformed."

        if summary_content and not summary_content.startswith("Error:"):
            logger.info(f"[LLM GenSummary] Successfully received response from LM Studio model.")
        # --- End Convenience API Usage ---

    except lms.LMStudioModelNotFoundError as e_model_nf:
        summary_content = f"LM Studio Model Not Found Error: {e_model_nf}. Ensure model '{lm_studio_model_identifier}' is available."
        logger.error(f"[LLM GenSummary] {summary_content}")
    except lms.LMStudioServerError as e_server:
        summary_content = f"LM Studio Server Error: {e_server}. Check LM Studio server status."
        logger.error(f"[LLM GenSummary] {summary_content}")
    except lms.LMStudioPredictionError as e_pred:
        summary_content = f"LM Studio Prediction Error: {e_pred}."
        logger.error(f"[LLM GenSummary] {summary_content}")
    except lms.LMStudioClientError as e_client:
        summary_content = f"LM Studio Client Error: {e_client}."
        logger.error(f"[LLM GenSummary] {summary_content}")
    except lms.LMStudioError as e_lms: # Generic LMStudio base error
        summary_content = f"Generic LM Studio Error: {e_lms}. Model: '{lm_studio_model_identifier}'."
        logger.error(f"[LLM GenSummary] {summary_content}")
    except Exception as e_llm:
        summary_content = f"Unexpected error during LLM interaction: {e_llm}."
        logger.error(f"[LLM GenSummary] Unexpected LLM interaction error: {e_llm}", exc_info=True)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        logger.info(f"[LLM GenSummary] LLM output saved to: {output_path}")
        
        # Check if the summary indicates an error before returning the path
        if "failed" in summary_content.lower() or "error" in summary_content.lower() or "ensure lm studio" in summary_content.lower():
            logger.warning(f"[LLM GenSummary] LLM interaction seems to have failed. Content: '{summary_content[:200]}...'")
            return None # Indicate failure to the caller more clearly
        return output_path
        
    except IOError as e_write:
        logger.error(f"[LLM GenSummary] Failed to write summary file to {output_path}: {e_write}")
        return None
    except Exception as e_final_write:
        logger.error(f"[LLM GenSummary] Unexpected error writing final summary to {output_path}: {e_final_write}")
        return None 