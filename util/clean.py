import settings
import os

def clean():
    filelist = [f for f in os.listdir(settings.OUTPUT_FOLDER) if os.path.isfile(os.path.join(settings.OUTPUT_FOLDER, f))]
    for f in filelist:
        os.remove(os.path.join(settings.OUTPUT_FOLDER, f))