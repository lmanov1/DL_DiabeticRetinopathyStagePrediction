import os
import sys
import re
from datetime import datetime

def save_terminal_output_to_file(output_dir="logs", filename_prefix="terminal_out", mode="w"):
    """
    Redirects terminal output to a file with a timestamp temporarily,
    then restores output back to the terminal.

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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # Construct the file path
    filename = f"{filename_prefix}_{timestamp}.txt"
    file_path = os.path.join(universal_path, filename)

    # Backup the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # Redirect output to file
        with open(file_path, mode) as f:
            sys.stdout = f
            sys.stderr = f
            print(f"Saving terminal output to file: {file_path}")
            # You can add any specific output you want here while in file redirection.

    except Exception as e:
        # Print the error to original stdout in case of failure
        original_stdout.write(f"Error redirecting output: {e}\n")

    finally:
        # Restore the original stdout and stderr to continue terminal printing
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f" ===> Terminal output restored. File saved at: {file_path}")

# Example usage
# save_terminal_output_to_file()


#delete later on
# def save_terminal_output_to_file(output_dir="logs", filename_prefix="terminal_out", mode="w"):
#     """
#     Redirects terminal output to a file with a timestamp.
#
#     Args:
#         output_dir (str): The directory where the output file will be saved.
#         filename_prefix (str): The prefix for the output filename.
#         mode (str): The file mode ('a' for append or 'w' for overwrite).
#     """
#     # Remove the filename using regex
#     new_output_dir = re.sub(r'\\[^\\]+$', '', output_dir)
#
#     # Convert to OS-independent format
#     universal_path = os.path.normpath(new_output_dir)
#
#     print(" ===> Saving terminal output to Directory: ", universal_path)
#     # Ensure the output directory exists
#     if not os.path.exists(universal_path):
#         os.makedirs(universal_path)
#
#     # Create a timestamp for the filename
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M')
#
#     # Construct the file path
#     filename = f"{filename_prefix}_{timestamp}.txt"
#     file_path = os.path.join(universal_path, filename)
#
#     # Try opening the file and redirecting output
#     try:
#         sys.stdout = open(file_path, mode)
#         sys.stderr = open(file_path, mode)  # Redirect stderr as well
#         print(f"Saving terminal output to file: {file_path}")
#     except Exception as e:
#         print(f"Error redirecting output: {e}")

# Example usage
# save_terminal_output_to_file()
# Original path
# path = "C:\\Users\\DELL\\Documents\\GitHub\\DL_DiabeticRetinopathyStagePrediction\\code\\data\\output\\trainLabels19_trained_model.pth"
#
# save_terminal_output_to_file(path)




