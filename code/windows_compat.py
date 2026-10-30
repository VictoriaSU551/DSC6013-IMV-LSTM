"""POSIX flock compatibility for the unchanged audited Linux training modules.

Only installed on Windows. The byte-range lock is released on close/process
exit; both the supervisor and workers lock byte zero of the same lock file.
"""
import os
import sys

def install():
    if os.name!='nt' or 'fcntl' in sys.modules:return
    import msvcrt
    import time
    import types
    module=types.ModuleType('fcntl')
    module.LOCK_SH=1;module.LOCK_EX=2;module.LOCK_NB=4;module.LOCK_UN=8
    def flock(file,operation):
        fd=file if isinstance(file,int) else file.fileno()
        offset=os.lseek(fd,0,os.SEEK_CUR)
        try:
            os.lseek(fd,0,os.SEEK_SET)
            if operation & module.LOCK_UN:
                msvcrt.locking(fd,msvcrt.LK_UNLCK,1);return
            while True:
                try:
                    mode=msvcrt.LK_NBRLCK if operation & module.LOCK_SH else msvcrt.LK_NBLCK
                    msvcrt.locking(fd,mode,1);return
                except OSError as exc:
                    if operation & module.LOCK_NB:raise BlockingIOError(str(exc)) from exc
                    time.sleep(.05)
        finally:os.lseek(fd,offset,os.SEEK_SET)
    module.flock=flock
    sys.modules['fcntl']=module
