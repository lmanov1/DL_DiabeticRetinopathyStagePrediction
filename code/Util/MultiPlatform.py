import os
import platform


def get_path_separator():
        system = platform.system()
        if system == 'Linux' or system == 'Darwin':  # Darwin is macOS
            separator = '/'
        elif system == 'Windows':
            separator = '\\'
        else:
            raise EnvironmentError("Unsupported operating system")
        
        print(f"Running on {system}. Path separator: {separator}")
        return separator
    
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