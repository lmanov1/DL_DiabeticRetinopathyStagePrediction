import os
import sys
import re
import io
from datetime import datetime

# Create a global buffer to capture terminal output
buffer = io.StringIO()

# Function to save the captured output to a file
def save_terminal_output_to_file(output_dir="logs", filename_prefix="terminal_out", mode="w"):
    """
    Saves the captured terminal output to a file with a timestamp.

    Args:
        output_dir (str): The directory where the output file will be saved.
        filename_prefix (str): The prefix for the output filename.
        mode (str): The file mode ('a' for append or 'w' for overwrite).
    """
    # Remove the filename using regex
    new_output_dir = re.sub(r'\\[^\\]+$', '', output_dir)

    # Convert to OS-independent format
    universal_path = os.path.normpath(new_output_dir)

    print(" ===> Saving terminal output to Directory: ", universal_path)

    # Ensure the output directory exists
    if not os.path.exists(universal_path):
        os.makedirs(universal_path)

    # Create a timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m')
    filename = f"{filename_prefix}_{timestamp}.txt"
    file_path = os.path.join(universal_path, filename)

    # Write buffer content to file
    with open(file_path, mode) as f:
        f.write(buffer.getvalue())  # Write captured output to file

    print(f" ===> Terminal output saved to: {file_path}")


# Function to perform actions before saving
def pre_save_actions():
    # Step 1: Backup original stdout and stderr
    original_stdout = sys.stdout  # Backup original stdout
    original_stderr = sys.stderr  # Backup original stderr

    # Step 2: Redirect stdout and stderr to the buffer
    sys.stdout = buffer  # Redirect stdout to buffer
    sys.stderr = buffer  # Redirect stderr to buffer

    print("Performing pre-save actions...")  # Example output to buffer

    return original_stdout, original_stderr  # Return backups for later restoration


# Function to perform actions after saving
def post_save_actions(original_stdout, original_stderr):
    # Restore original stdout and stderr
    sys.stdout = original_stdout  # Restore original stdout
    sys.stderr = original_stderr  # Restore original stderr

    print("Performing post-save actions...")  # Example output to terminal
