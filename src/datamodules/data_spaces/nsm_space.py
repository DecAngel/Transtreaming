import os
import sys

import pickle
from dataclasses import dataclass

import math
import re
import subprocess
import multiprocessing.shared_memory as sm
from multiprocessing.managers import SharedMemoryManager, dispatch
from multiprocessing import resource_tracker, Process
from typing import Tuple, Optional, Protocol, Dict, Literal, Any
import torch


from src.primitives.datamodule import BaseDataSpace


@dataclass
class TensorInfo:
    dtype: torch.dtype
    shape: torch.Size


class NSMDataSpace(SharedMemoryManager, BaseDataSpace):
    def __init__(
            self,
            prefix: str = '',
            method: Literal['connect', 'start', 'default', 'none'] = 'default',
            address: Tuple[str, int] = ('127.0.0.1', 16283),
            authkey: Optional[bytes] = b'iva\xad\xd5~\x08\x8f\r%'
    ):
        super().__init__(address, authkey, 'pickle', None)
        self.prefix = prefix
        self.safe_margin = 1 * (2 ** 20)
        self._close_list = []
        self._serving = False

        if method == 'default':
            try:
                self.connect()
            except (TimeoutError, ConnectionResetError, ConnectionRefusedError):
                self.start()
        elif method == 'connect':
            self.connect()
        elif method == 'start':
            self.start()

    def start(self, initializer = None, initargs = ()):
        super().start(initializer, initargs)
        self._serving = True

    def __del__(self):
        for s in self._close_list:
            s.close()

    def _get_shm_limit(self) -> int:
        """Get available shared memory in bytes.

        """
        if sys.platform == 'linux':
            proc = subprocess.Popen(
                'df --sync',
                stdout=subprocess.PIPE, shell=True, text=True,
            )
            x = proc.communicate()[0]
            rem = int(re.findall(r'(\d*)\s*\d*%\s*/dev/shm', x)[0]) * (2 ** 10)
        elif sys.platform == 'win32':
            import win32api
            rem = win32api.GlobalMemoryStatusEx()['AvailPhys']
        else:
            raise ValueError('Unsupported platform')
        return rem - self.safe_margin

    def _get_path(self, name: str) -> str:
        return name if self.prefix == '' else f'{self.prefix}.{name}'

    def _has_nsm(self, name: str) -> bool:
        try:
            sms = sm.SharedMemory(name=self._get_path(name), create=False)
            if not self._serving:
                resource_tracker.unregister(sms._name, 'shared_memory')
            sms.close()
        except (OSError, FileNotFoundError):
            return False
        else:
            return True

    def _get_nsm(self, name: str) -> sm.SharedMemory:
        """Returns a new SharedMemory instance with the specified size in
                    bytes, to be tracked by the manager."""
        # print(f'get {self._get_path(name)} from {os.getpid()}')
        sms = sm.SharedMemory(name=self._get_path(name), create=False)

        # manual close and unlink
        if not self._serving:
            resource_tracker.unregister(sms._name, 'shared_memory')
        sms.__del__ = None
        self._close_list.append(sms)
        return sms

    def _rm_nsm(self, name: str):
        path = self._get_path(name)
        self._close_list = filter(lambda p: p is not None, [sms if sms.name != path else sms.close() for sms in self._close_list])

        with self._Client(self._address, authkey=self._authkey) as conn:
            try:
                dispatch(conn, None, 'release_segment', (path,))
            except BaseException as e:
                raise e

    def _set_nsm(self, name: str, size: int) -> sm.SharedMemory:
        """Returns a new SharedMemory instance with the specified size in
                            bytes, to be tracked by the manager."""
        # print(f'set {self._get_path(name)} from {os.getpid()}')

        if self._get_shm_limit() < size + self.safe_margin:
            raise MemoryError(f'Insufficient shm memory. '
                              f'{size} bytes required but '
                              f'only {self._get_shm_limit() - self.safe_margin} bytes available.')

        sms = sm.SharedMemory(name=self._get_path(name), create=True, size=size)
        if not self._serving:
            resource_tracker.unregister(sms._name, 'shared_memory')
        sms.__del__ = None
        self._close_list.append(sms)
        with self._Client(self._address, authkey=self._authkey) as conn:
            try:
                dispatch(conn, None, 'track_segment', (sms.name,))
            except BaseException as e:
                sms.unlink()
                raise e
        return sms

    def _get(self, prefix: str, obj: Any) -> Any:
        if isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self._get(f'{prefix}.{k}', v)
            return type(obj)(res)
        elif isinstance(obj, (list, tuple)):
            res = []
            for i, item in enumerate(obj):
                res.append(self._get(f'{prefix}.{i}', item))
            return type(obj)(res)
        elif isinstance(obj, TensorInfo):
            sms = self._get_nsm(prefix)
            return torch.frombuffer(sms.buf, dtype=obj.dtype, count=math.prod(obj.shape)).reshape(obj.shape)
        else:
            return obj

    def _set(self, prefix: str, obj: Any) -> Any:
        if isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self._set(f'{prefix}.{k}', v)
            return type(obj)(res)
        elif isinstance(obj, (list, tuple)):
            res = []
            for i, item in enumerate(obj):
                res.append(self._set(f'{prefix}.{i}', item))
            return type(obj)(res)
        elif isinstance(obj, torch.Tensor):
            if not self._has_nsm(prefix):
                sms = self._set_nsm(prefix, obj.element_size() * math.prod(obj.shape))
                t = torch.frombuffer(sms.buf, dtype=obj.dtype, count=math.prod(obj.shape)).reshape(obj.shape)
                t.copy_(obj)
            return TensorInfo(obj.dtype, obj.shape)
        else:
            return obj

    def __contains__(self, name: str) -> bool:
        return self._has_nsm(f'{name}._meta')

    def __getitem__(self, name: str) -> Dict[str, Any]:
        sms = self._get_nsm(f'{name}._meta')
        meta = pickle.loads(sms.buf)
        return self._get(name, meta)

    def __setitem__(self, name: str, d: Dict[str, Any]):
        meta = self._set(name, d)
        meta_dump = pickle.dumps(meta)
        if self._has_nsm(f'{name}._meta'):
            self._rm_nsm(f'{name}._meta')
        sms = self._set_nsm(f'{name}._meta', len(meta_dump))
        sms.buf[:] = meta_dump


if __name__ == '__main__':
    def child1():
        nsmm = NSMDataSpace(prefix='nsmm', method='connect')
        nsmm['a'] = {
            'b': torch.arange(0, 1000000000, dtype=torch.float32).reshape(5, 200000000),
            'c': [1, 2, 3, 'str1', 'str2'],
            'd': torch.ones(7, 8, dtype=torch.int32),
        }
        print(f'{os.getpid()} finished')

    def child2():
        nsmm = NSMDataSpace(prefix='nsmm', method='connect')
        e = {
            'f': torch.arange(0, 1000000000, dtype=torch.float32).reshape(5, 200000000),
            'g': [1, 2, 3, 'str1', 'str2'],
            'h': torch.ones(7, 8, dtype=torch.int32),
        }
        print('a' in nsmm)
        nsmm['e'] = e
        a = nsmm['a']
        b = nsmm['a']
        print(a['b'].shape)
        print(a['b'][3, 4])
        print(e['f'].shape)
        print(e['f'][3, 4])
        print(a['b'][3, 4] == e['f'][3, 4])
        a['b'][3, 4] = -1
        print(a['b'][3, 4] == e['f'][3, 4])
        print(a['b'][3, 4] == b['b'][3, 4])
        print(f'{os.getpid()} finished')


    nsmm = NSMDataSpace(prefix='nsmm', method='start')
    p1 = Process(target=child1)
    p1.start()
    p1.join()

    p2 = Process(target=child2)
    p2.start()
    p2.join()
    nsmm.shutdown()
    nsmm.join()
