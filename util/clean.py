import settings
import os

def clean():
    filelist = [f for f in os.listdir(settings.OUTPUT_FOLDER) if f.endswith('mmap')]
    for f in filelist:
        os.remove(os.path.join(settings.OUTPUT_FOLDER, f))
