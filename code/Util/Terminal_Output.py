import os
import sys
from datetime import datetime

def save_terminal_output_to_file(output_dir="logs", filename_prefix="output", mode="w"):
    """
    Redirects terminal output to a file with a timestamp.

    Args:
        output_dir (str): The directory where the output file will be saved.
        filename_prefix (str): The prefix for the output filename.
        mode (str): The file mode ('a' for append or 'w' for overwrite).
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Construct the file path
    filename = f"{filename_prefix}_{timestamp}.txt"
    file_path = os.path.join(output_dir, filename)

    # Try opening the file and redirecting output
    try:
        sys.stdout = open(file_path, mode)
        sys.stderr = open(file_path, mode)  # Redirect stderr as well
        print(f"Saving terminal output to file: {file_path}")
    except Exception as e:
        print(f"Error redirecting output: {e}")

# Example usage
# save_terminal_output_to_file()
