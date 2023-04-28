import os 
import logging
import logging.config
from pathlib import Path
import urllib
import platform
import contextlib
import threading

VERBOSE = str(os.getenv('Prompt-Can-Anything_VERBOSE', True)).lower() == 'true'  # global verbose mode
AUTOINSTALL = str(os.getenv('Prompt-Can-Anything_AUTOINSTALL', True)).lower() == 'true'  # global auto-install 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # 
IMAGENET_MEAN= [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
COCO_MEAN = [0.5, 0.5, 0.5]
COCO_STD = [0.5, 0.5, 0.5]
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp' ,'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

def set_logging(name=None, verbose=VERBOSE):
    # Sets level and returns    
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.WARNING
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)
    
LOGGING_NAME='Prompt-Can-Anything'
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally 

def write_categories(cls_name, file_path):
    # 打开TXT文件，以写入模式覆盖文件内容
    with open(file_path, 'a') as f:
        # 将列表元素写入TXT文件，每个元素都写入新的一行
        f.write(cls_name+"\n")
        # 强制刷新缓存，确保文件已经被完全写入
        f.flush()

        
def threaded(func):
    # Multi-threads a target function and returns thread. Usage: @threaded decorator
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper

class TryExcept(contextlib.ContextDecorator):
    # reference:YOLOv8 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg='', verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True
    
def check_suffix(file=None, suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"

def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    import socket
    with contextlib.suppress(Exception):
        host = socket.gethostbyname('www.github.com')
        socket.create_connection((host, 80), timeout=2)
        return True
    return False

def url2file(url):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return Path(urllib.parse.unquote(url)).name.split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?aut

def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str
def clean_url(url):
     
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return urllib.parse.unquote(url).split('?')[0]  # '%2F' to '/', split https://url.com/file.txt?auth

def check_requirements(requirements=ROOT / 'requirements.txt', exclude=(), install=True, cmds=()):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr('red', 'bold', 'requirements:')
    #check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for i, r in enumerate(requirements):
        try:
            pkg.require(r)
        except Exception:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required "
            if install and AUTOINSTALL:  # check environment variable
                LOGGER.info(f"{s}, attempting auto-update...")
                try:
                    #assert check_online(), f"'pip install {r}' skipped (offline)"
                    LOGGER.info(check_output(f"pip install '{r}' {cmds[i] if cmds else ''}", shell=True).decode())
                    n += 1
                except Exception as e:
                    LOGGER.warning(f'{prefix} ❌ {e}')
            else:
                LOGGER.info(f'{s}. Please install and rerun your command.')

    if n:  # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n" \
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        LOGGER.info(s)