import os
import subprocess
import traceback
import logging
import gradio as gr
from threading import Thread, Lock
from time import sleep

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]: %(message)s")

# Shared variables for process tracking
output_log = ""
output_lock = Lock()
current_process = None
process_lock = Lock()  # Ensure thread-safe access to current_process


def run_fap_mix_plus(url, uploaded_files, output_dir):
    """Run the fapMixPlus.py script using subprocess."""
    global current_process, output_log
    logging.info("Starting audio processing...")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare input and arguments
    args = ["python", "fapMixPlus.py", "--output_dir", output_dir]

    if url:
        args += ["--url", url]
    elif uploaded_files:
        input_dir = os.path.join(output_dir, "uploaded")
        os.makedirs(input_dir, exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join(input_dir, os.path.basename(uploaded_file.name))
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
        args.append(input_dir)
    else:
        logging.error("No input provided.")
        with output_lock:
            output_log += "Error: No input provided.\n"
        return "Error: No input provided."

    def subprocess_thread():
        """Run the subprocess in a separate thread to capture output."""
        global current_process, output_log
        try:
            logging.debug(f"Running subprocess with arguments: {args}")
            with process_lock:
                current_process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

            # Capture real-time output
            for line in current_process.stdout:
                logging.info(line.strip())
                with output_lock:
                    output_log += line

            for line in current_process.stderr:
                logging.error(line.strip())
                with output_lock:
                    output_log += line

            # Wait for the process to finish
            current_process.wait()
            if current_process.returncode != 0:
                with output_lock:
                    output_log += f"Script failed with return code {current_process.returncode}.\n"

        except Exception as e:
            error_message = f"An error occurred:\n{traceback.format_exc()}"
            logging.error(error_message)
            with output_lock:
                output_log += error_message
        finally:
            with process_lock:
                current_process = None

    # Start the thread
    Thread(target=subprocess_thread).start()
    return "Process started. Logs will update shortly."


def cancel_process():
    """Cancel the running subprocess."""
    global current_process
    with process_lock:
        if current_process is not None:
            logging.info("Cancelling the subprocess...")
            current_process.terminate()
            try:
                current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                current_process.kill()
            finally:
                current_process = None
                return "Process cancelled."
        else:
            return "No process is currently running."


def fetch_logs():
    """Fetch the latest logs."""
    with output_lock:
        return output_log


def find_latest_zip(output_dir):
    """Find the latest ZIP file in the most recently created subfolder."""
    try:
        # List all subdirectories in the output directory
        subfolders = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]

        # Sort subfolders by creation time (newest first)
        subfolders.sort(key=os.path.getmtime, reverse=True)

        # Look for ZIP files in the most recent subfolder
        for subfolder in subfolders:
            zip_files = [f for f in os.listdir(subfolder) if f.endswith(".zip")]
            if zip_files:
                # Return the full path to the first ZIP file found
                return os.path.join(subfolder, zip_files[0])

        # No ZIP files found
        return None
    except Exception as e:
        logging.error(f"Error finding latest ZIP file: {e}")
        return None


# Gradio app definition
with gr.Blocks() as app:
    gr.Markdown("# Audio Processing App")
    url = gr.Textbox(label="Audio URL", placeholder="Enter the URL to download audio (optional)")
    upload_dir = gr.File(label="Upload Files", file_types=[".wav", ".mp3"], file_count="multiple")
    output_dir = gr.Textbox(label="Output Directory", value="output", placeholder="Enter output directory")
    run_button = gr.Button("Process Audio")
    cancel_button = gr.Button("Cancel Process")
    result = gr.Textbox(label="Result", interactive=False, lines=15, max_lines=30, value="")
    download_button = gr.Button("Download ZIP")
    download_link = gr.File(interactive=False)

    # Periodic log updates
    def refresh_logs():
        """Fetch and return the latest logs."""
        return fetch_logs()

    run_button.click(run_fap_mix_plus, [url, upload_dir, output_dir], [result])
    cancel_button.click(cancel_process, [], result)
    download_button.click(lambda output_dir: find_latest_zip(output_dir), inputs=[output_dir], outputs=[download_link])
    gr.Button("Refresh Logs").click(refresh_logs, [], result)  # Add a button for manual refresh

app.launch()
