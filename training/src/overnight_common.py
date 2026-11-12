from pathlib import Path
import json,os,datetime,ctypes,hashlib


def utcnow():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(temporary,path)


def process_identity():
    import psutil
    process=psutil.Process();parents=[]
    for parent in process.parents():
        try:parents.append({'pid':parent.pid,'name':parent.name()})
        except psutil.Error:pass
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.IsProcessInJob.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
    flag=ctypes.c_int();kernel.IsProcessInJob(kernel.GetCurrentProcess(),None,ctypes.byref(flag))
    return {'pid':os.getpid(),'parents':parents,'in_windows_job':bool(flag.value)}


def code_hash(root):
    paths=sorted((Path(root)/'src').glob('*.py'))
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes() for p in paths)).hexdigest()
