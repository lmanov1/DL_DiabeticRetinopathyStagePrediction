import os
import platform


def get_path_separator():
        return os.path.sep
        # system = platform.system()
        # if system == 'Linux' or system == 'Darwin':  # Darwin is macOS
        #     separator = '/'
        # elif system == 'Windows':
        #     separator = '\\'
        # else:
        #     raise EnvironmentError("Unsupported operating system")
        
        # print(f"Running on {system}. Path separator: {separator}")
        # return separator
    
def get_home_directory():    
    system = platform.system()
    if system == 'Linux' or system == 'Darwin':  # Darwin is macOS
        home_dir = os.environ.get('HOME')        
    elif system == 'Windows':
        home_dir = os.environ.get('USERPROFILE')
    else:
        raise EnvironmentError("Unsupported operating system")
    
    print(f"Running on {system}. Home directory: {home_dir}")
    return home_dir


def print_time(start_secs, end_secs, title=""):
    elapsed_time = end_secs - start_secs
    days = elapsed_time // (24 * 3600)
    elapsed_time = elapsed_time % (24 * 3600)
    hours = elapsed_time // 3600
    elapsed_time %= 3600
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    print(f"==> {title}: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds")
    return days, hours, minutes, seconds
