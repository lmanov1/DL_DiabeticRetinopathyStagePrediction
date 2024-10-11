# DL_DiabeticRetinopathyStagePrediction
This repo hosts a final DL project conducted as a part of data scientist certification at BIU



Getting started
1. Prerequerements: 
    - Be sure you have poetry installed in your environment    
    - If you have GPU , install nvidia-smi cli      
2. clone the repo  , go to the cloned directory   
    `git clone git@github.com:lmanov1/DL_DiabeticRetinopathyStagePrediction.git`
3. Run setup: this should detect GPU and install supporting python system libraries (unsupported by poetry) like CUDA     
    `python3 code/Util/check_hardware_and_install.py`    
4. Run `poetry install` (just once)     
    Don't worry , without available GPU (and CUDA) , tensorflow, torch and rest of libraries leveraging GPU will automatically use the CPU.     
5. Run `poetry shell`
6.  About Kaggle API  
    We use Kaggle API to download datasets from Kaggle.           
    To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com     
    Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location appropriate for your operating system:

    - Linux: `$XDG_CONFIG_HOME/kaggle/kaggle.json` (defaults to `~/.config/kaggle/kaggle.json`). The path `~/.kaggle/kaggle.json` which was used by older versions of the tool is also still supported.
     `chmod 600 ~/.config/kaggle/kaggle.json` - no read access for other users.    
    - Windows: C:\Users\<Windows-username>\.kaggle\kaggle.json - you can check the exact location, sans drive, with echo %HOMEPATH%.        
    - Other: ~/.kaggle/kaggle.json     

    - You can define a shell environment variable KAGGLE_CONFIG_DIR to change this location to $KAGGLE_CONFIG_DIR/kaggle.json (on Windows it will be %KAGGLE_CONFIG_DIR%\kaggle.json).
    
    - You can also choose to export your Kaggle username and token to the environment:
    export KAGGLE_USERNAME=datadinosaur
    export KAGGLE_KEY=xxxxxxxxxxxxxx
    In addition, you can export any other configuration value that normally would be in the kaggle.json in the format 'KAGGLE_' (note uppercase).

7. Now you all set and can run project logics , for example       
`poetry run python3 /code/main.py`      
`poetry run python3 code/data/Dataloader.py`
