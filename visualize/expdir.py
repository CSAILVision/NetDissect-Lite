import glob
import os
import re
import shutil

# Add basestring for python3
try:
    basestring
except NameError:
    basestring = str

class ExperimentDirectory:
    '''
    Management of the files and filenames within a dissection.
    '''

    def __init__(self, directory):
        self.directory = os.path.expanduser(directory)

    def basename(self):
        return os.path.basename(self.directory)

    def filename(self, name=None, last=True, decimal=False,
            blob=None, part=None, directory=None, aspair=False):

        # Handle arrays of names
        if isinstance(name, basestring):
            name = [name]
        elif name is None:
            name = []
        # Expand out dissection directory.
        path = [self.directory]
        if directory is not None:
            path.append(directory)
        # Insert part name into filename if specified.
        if part is not None:
            name.insert(0, part)
        # Insert blob name into filename if specified.
        if blob is not None:
            name.insert(0, fn_safe(blob))
        # Assemble filename
        path.append('-'.join(name))
        fn = os.path.join(*path)
        # Expand out numbered globs.
        if '*' in fn:
            n, fn = numbered_glob(fn, last=last, decimal=decimal)
            if aspair:
                return (n, fn)
        return fn

    def has(self, name=None, last=True, decimal=False,
            blob=None, part=None, directory=None, aspair=False):
        return os.path.exists(self.filename(name=name, last=last,
            decimal=decimal, blob=blob, part=part, directory=directory))

    def glob_number(self, name, last=True, decimal=False, blob=None, part=None):
        '''
        For a globbed filename, returns the matching number rather than
        the matching filename.
        '''
        n, fn = self.filename(name, last=last, decimal=decimal,
                blob=blob, part=part, aspair=True)
        return n

    def ensure_dir(self, *args):
        '''
        Creates a directory if it does not yet exist.
        '''
        dirname = os.path.join(*([self.directory] + list(args)))
        try:
            os.makedirs(dirname)
        except OSError:
            pass

    def remove_all(self, *args):
        '''
        Removes all files that match a given pattern.
        '''
        try:
            for c in glob.glob(os.path.join(*([self.directory] + args))):
                os.remove(f)
        except:
            pass

    ###################################################################
    # html/[blob][-part].html, defaulting to index.html
    ###################################################################
    def html_filename(self, blob=None, part=None, **kwargs):
        if blob is None and part is None:
            blob = 'index'
        result = self.filename('html/%s.html' % (
            '-'.join(filter(None, [fn_safe(blob), part])),), **kwargs)
        return result

    def save_html(self, html, blob=None, part=None, fieldnames=None):
        filename = self.html_filename(blob=blob, part=part)
        wrappers = ['html', 'body']
        with open(filename, 'w') as f:
            f.write('<!doctype html>')
            suffix = []
            for w in wrappers:
                if ('<' + w) not in html:
                    f.write('<%s>\n' % w)
                    suffix.insert(0, '</%s>\n' % w)
            f.write(html)
            f.write('\n'.join(suffix))

    ###################################################################
    # [blob][-part]/ directory
    ###################################################################

    def working_dir(self, blob=None, part=None):
        result = '-'.join(filter(None, [fn_safe(blob), part]))
        self.ensure_dir(result)
        return result

    def remove_dir(self, filename):
        shutil.rmtree(self.filename(filename), ignore_errors=True)

def fn_safe(blob, dotfree=False):
    '''Sometimes a blob name will have delimiters in it which are not
    filename safe.  Fix these by turning them into hyphens.'''
    if blob is None:
        return None
    if dotfree:
        return re.sub('[\.-/#?*!\s]+', '-', blob).strip('-')
    else:
        return re.sub('[-/#?*!\s]+', '-', blob).strip('-')

def numbered_glob(pattern, last=True, decimal=False, every=False):
    '''Given a globbed file_*_pattern, returns a pair of a matching
    filename along with a matching number.'''
    repat = r'(\d+(?:\.\d*)?)'.join(
            re.escape(s) for s in pattern.split('*', 1))
    best_fn = None
    best_n = None
    all_results = []
    for c in glob.glob(pattern):
        m = re.match(repat, c)
        if m:
            if decimal:
                if '.' in m.group(1):
                    n = float(m.group(1))
                else:
                    n = float('0.' + m.group(1))
            else:
                n = int(m.group(1).strip('.'))
            all_results.append((c, n))
            if best_n is None or (best_n < n) == last:
                best_n = n
                best_fn = c
    if every:
        return all_results
    if best_fn is None:
        raise IOError(pattern)
    return best_n, best_fn

