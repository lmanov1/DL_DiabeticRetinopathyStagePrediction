import sys
import os
from datetime import datetime


def save_terminal_output_to_file(output_dir="logs", filename_prefix="output", mode="a"):
    """
    Redirects terminal output to a file with a timestamp.

    Args:
        output_dir (str): The directory where the output file will be saved.
        filename_prefix (str): The prefix for the output filename.
        mode (str): The file mode ('a' for append or 'w' for overwrite).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Construct the file path
    filename = f"{filename_prefix}_{timestamp}.txt"
    file_path = os.path.join(output_dir, filename)

    # Open the file and redirect the output
    sys.stdout = open(file_path, mode)
    sys.stderr = open(file_path, mode)  # Redirect stderr as well, if needed
    print(f"Saving terminal output to file: {file_path}")

# Call this function at the start of the script where you want to capture output
# For example:
#

