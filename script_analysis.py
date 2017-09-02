import os
import json
from functools import wraps
import errno
import os
import signal
import Timbral_Brightness as bright
import Timbral_Roughness as rough
import Timbral_Metallic as metal
import Timbral_Reverb as reverb
import Timbral_Hardness as hard

"""
LAUNCH ANALYSIS:
> python script_analysis.py


BASH COMMAND FOR CONVERSION: (already done in the folder sound_files)
Still there were mp3 file without "mp3" in the name. Did some manually.
A better way to do it would be to use a function to know the compression format
Convert all files that have .mp3 in the name:
> for f in *.mp3*; do ffmpeg -i "$f" -acodec pcm_u8 -ar 44100  "$f.wav" ;  done

Remove .wav at the end of the file:
> for f in *.wav; do mv "$f" "${f%.wav}"; done

"""

def process_analysis(analysis_method, file_name, result_dict, fail_list):
    """
    Start an analysis and store data.
    """
    fs_id = split(file_name)
    fname = 'sound_files/' + file_name
    try:
         result_dict[fs_id] = wrap_method(analysis_method, fname)
    except:
        print '\n fail at %s for sound %s \n' % (analysis_method.func_name, file_name)
        fail_list.append((analysis_method.func_name, file_name))

def split(s):
    """
    Split a string with -#- and return the last string.
    Used for getting Freesound id of sounds.
    """
    return s.split(' -#- ')[-1]
    
class ProgressBar:
    """
    Progress bar
    """
    def __init__ (self, valmax, maxbar, title):
        if valmax == 0:  valmax = 1
        if maxbar > 200: maxbar = 200
        self.valmax = valmax
        self.maxbar = maxbar
        self.title  = title
        print ''

    def update(self, val):
        import sys
        # format
        if val > self.valmax: val = self.valmax

        # process
        perc  = round((float(val) / float(self.valmax)) * 100)
        scale = 100.0 / float(self.maxbar)
        bar   = int(perc / scale)

        # render
        out = '\r %20s [%s%s] %3d / %3d' % (self.title, '=' * bar, ' ' * (self.maxbar - bar), val, self.valmax)
        sys.stdout.write(out)
        sys.stdout.flush()

        
class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """
    Decorator for raising exception after a method takes to long to compute.
    Some methods fail and get stuck...
    """
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator

@timeout(30)
def wrap_method(analysis_method, fname):
    return analysis_method(fname)


if __name__ == '__main__':   
    files = os.listdir('sound_files/')
    nb_sounds = len(files)
    bar = ProgressBar(nb_sounds, 30, 'Analysing ...')

    bright_analysis = {}
    rough_analysis = {}
    metal_analysis = {}
    reverb_analysis = {}
    hard_analysis = {}
    
    list_fails = []
    
    for idx, fname in enumerate(files):
        bar.update(idx+1)     
        process_analysis(bright.timbral_brightness, fname, bright_analysis, list_fails)
        process_analysis(rough.timbral_roughness, fname, rough_analysis, list_fails)
        process_analysis(metal.timbral_metallic, fname, metal_analysis, list_fails)
        process_analysis(reverb.timbral_reverb, fname, reverb_analysis, list_fails)
        process_analysis(hard.timbral_hardness, fname, hard_analysis, list_fails)
        
    json.dump(bright_analysis, open('bright_analysis.json','w'))
    json.dump(rough_analysis, open('rough_analysis.json','w'))
    json.dump(metal_analysis, open('metal_analysis.json','w'))
    json.dump(reverb_analysis, open('reverb_analysis.json','w'))
    json.dump(hard_analysis, open('hard_analysis.json','w'))
    json.dump(list_fails, open('list_fails.json', 'w'))

    