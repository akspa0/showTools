import logging
from pathlib import Path
import json
import os
import lmstudio as lms

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
    """
    if not clap_events_data or not isinstance(clap_events_data, dict):
        logger.warning("[LLM Summary] CLAP events data is missing or not in expected dict format.")
        return "No sound events detected or data is malformed."

    formatted_events = []
    # Check for the main 'clap_events' list, which is the structure output by ClapAnnotator wrapper
    events_list = clap_events_data.get("clap_events")
    
    # If "clap_events" key doesn't exist, maybe the data is a direct list or another structure?
    # This part is an attempt to handle slightly different structures gracefully.
    if events_list is None:
        # If the root is a list, assume it's the events_list
        if isinstance(clap_events_data, list):
            events_list = clap_events_data
            logger.warning("[LLM Summary] CLAP data was a direct list, not a dict with 'clap_events' key. Processing as list.")
        # If it's a dict but doesn't have "clap_events", look for other common top-level list keys
        elif isinstance(clap_events_data, dict):
            for key in ["events", "detections", "segments"]:
                if isinstance(clap_events_data.get(key), list):
                    events_list = clap_events_data.get(key)
                    logger.warning(f"[LLM Summary] CLAP data missing 'clap_events' key, but found list under '{key}'. Processing this list.")
                    break
            if events_list is None:
                logger.warning("[LLM Summary] CLAP data is a dict but does not contain a recognized events list (e.g., 'clap_events').")
                return "Sound event data found but structure not recognized or events list is missing."

    if not events_list: # if events_list is empty or became None
        return "No sound events detected."

    for event in events_list:
        if not isinstance(event, dict):
            logger.warning(f"[LLM Summary] Skipping non-dict event in CLAP data: {event}")
            continue
        label = event.get("label", "Unknown Event")
        start_time = event.get("start_time_seconds", "N/A")
        end_time = event.get("end_time_seconds", "N/A")
        confidence = event.get("confidence", "N/A")
        formatted_events.append(f"- Event: {label}, Start: {start_time}s, End: {end_time}s, Confidence: {confidence}")
    
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
    # Defaulting to a generic Llama 3 example, user should ensure this or their desired model is available in LM Studio
    lm_studio_model_identifier = config.get('lm_studio_model_identifier', 'NousResearch/Hermes-2-Pro-Llama-3-8B') 
    logger.info(f"[LLM Summary] Attempting to use LM Studio model: {lm_studio_model_identifier}")

    # Load and format transcript
    transcript_prompt_text = "Could not load transcript."
    if transcript_file_path and Path(transcript_file_path).exists():
        try:
            with open(transcript_file_path, 'r', encoding='utf-8') as f_trans:
                transcript_data = json.load(f_trans)
                transcript_prompt_text = _format_transcript_for_prompt(transcript_data)
        except Exception as e_read_trans:
            transcript_prompt_text = f"Error reading or parsing transcript: {e_read_trans}"
            logger.error(f"[LLM Summary] Error reading transcript file {transcript_file_path}: {e_read_trans}")
    else:
        logger.warning(f"[LLM Summary] Transcript file not found or not provided: {transcript_file_path}")

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
        logger.info(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] Initializing LM Studio client for model: {lm_studio_model_identifier}")
        # --- Refactored Client Usage ---
        # Default client, assumes LM Studio server at http://localhost:1234/v1
        # One could make base_url configurable if needed: base_url=config.get('lm_studio_base_url', 'http://localhost:1234/v1')
        client = lms.LMSClient() 
        logger.info(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] LMSClient initialized. Default base_url: {client.base_url}")

        messages = [
            {"role": "system", "content": "You are an AI assistant. Your task is to provide a concise summary of the following conversation transcript. Incorporate information about notable sound events if they provide relevant context."},
            {"role": "user", "content": f"Conversation Transcript:\n---\n{transcript_prompt_text}\n---\n\nNotable Sound Events Information:\n---\n{clap_events_prompt_text}\n---\n\nPlease generate a summary of the conversation:"}
        ]
        
        # Configurable parameters for the chat completion, e.g., temperature, max_tokens
        temperature = config.get('lm_studio_temperature', 0.7)
        max_tokens = config.get('lm_studio_max_tokens', 500) # Defaulting to a reasonable max for summaries

        logger.info(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] Sending request to LM Studio model: {lm_studio_model_identifier}. Temp: {temperature}, MaxTokens: {max_tokens}")
        
        completion = client.get_chat_completion(
            model=lm_studio_model_identifier,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            # stream=False # Default is False for get_chat_completion
        )
        
        # Extract response
        if completion and completion.get('choices') and isinstance(completion['choices'], list) and len(completion['choices']) > 0:
            message_content = completion['choices'][0].get('message', {}).get('content')
            if message_content:
                summary_content = message_content.strip()
                logger.info(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] Successfully received summary from LM Studio model.")
            else:
                summary_content = "LM Studio Error: Received empty message content from model."
                logger.error(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] {summary_content}. Full API response: {completion}")
        else:
            summary_content = f"LM Studio Error: Unexpected response structure from model '{lm_studio_model_identifier}'. Check LM Studio logs."
            logger.error(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] {summary_content}. Full API response: {completion}")
        # --- End Refactored Client Usage ---

    except lms.errors.APIConnectionError as e_conn:
        summary_content = f"LM Studio Connection Error: {e_conn}. Ensure LM Studio server is running at the expected address (likely http://localhost:1234/v1) and is accessible."
        logger.error(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] {summary_content}")
    except lms.errors.LMStudioSDKError as e_sdk: # Catch other SDK specific errors
        summary_content = f"LM Studio SDK Error: {e_sdk}. Model: '{lm_studio_model_identifier}'."
        logger.error(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] {summary_content}")
    except Exception as e_llm: # Broad exception for other errors
        # The original error message included the model identifier and a hint to check if it's loaded.
        # We should try to preserve that if possible, or make the new error equally informative.
        summary_content = f"LM Studio Error: An unexpected error occurred: {e_llm}. Ensure LM Studio is running, the model '{lm_studio_model_identifier}' is loaded/served, and the server is accessible."
        logger.error(f"[{pii_safe_file_prefix if pii_safe_file_prefix else 'LLM Summary'}] LM Studio client or model interaction error: {e_llm}", exc_info=True)

    try:
        with open(final_summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
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
        logger.info(f"[LLM GenSummary] Initializing LM Studio client.")
        client = lms.LMSClient() # Default client, assumes LM Studio server at localhost:1234
        
        logger.info(f"[LLM GenSummary] Sending request to LM Studio model: {lm_studio_model_identifier} with system prompt.")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_transcript_text}
        ]

        # Parameters for the chat completion
        # max_tokens for response, temperature, etc. can be added here if needed
        # And potentially made configurable via llm_config
        completion_params = {
            "model": lm_studio_model_identifier,
            "messages": messages,
            # "temperature": llm_config.get("temperature", 0.7), # Example
            # "max_tokens": llm_config.get("max_response_tokens", 1024) # Example for response length
        }
        
        # The lmstudio-python library documentation should be consulted for exact parameters 
        # related to context window if they exist at the client.chat.completions.create level.
        # Often, context window is a server-side model setting.
        logger.debug(f"[LLM GenSummary] Creating chat completion with params: {completion_params}")
        completion = client.chat.completions.create(**completion_params)
        
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            summary_content = completion.choices[0].message.content.strip()
            logger.info(f"[LLM GenSummary] Successfully received response from LM Studio model.")
        else:
            logger.error("[LLM GenSummary] LM Studio response was empty or malformed.")
            summary_content = "Error: LLM response was empty or malformed."

    except lms.exceptions.LMStudioSDKError as e_sdk:
        summary_content = f"LM Studio SDK Error: {e_sdk}. Ensure LM Studio is running, the model '{lm_studio_model_identifier}' is loaded/served, and network is configured."
        logger.error(f"[LLM GenSummary] LM Studio SDK error: {e_sdk}")
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