import re
import socket
import subprocess
import sys
import threading
from typing import Optional, List

from tensorboard.program import TensorBoard

from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def find_unused_port() -> int:
    # create a temporary socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # bind the sockets to random addresses
        s.bind(('', 0))
        # retrieve the port numbers that was allocated
        port = s.getsockname()[1]
    return port


def get_all_local_ip() -> List[str]:
    proc = subprocess.Popen(
        'ipconfig' if sys.platform.startswith('win') else 'ifconfig',
        stdout=subprocess.PIPE, shell=True, text=True,
    )
    x = proc.communicate()[0]
    return list(filter(
        lambda i: not i.endswith(('.0', '.255')),
        re.findall(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', x)
    ))


class TBService(TensorBoard):
    def __init__(self, log_dir: str, port: Optional[int] = None):
        super().__init__()
        if is_port_in_use(port):
            self.port = find_unused_port()
            logger.warning(f'Port {port} is occupied, reassign it to {self.port}.')
        else:
            self.port = port
        self.log_dir = log_dir
        self.configure(
            argv=[None, '--logdir', self.log_dir, '--bind_all', '--port', str(self.port)]
        )
        self.server = self._make_server()
        self.thread = threading.Thread(
            target=self.server.serve_forever, name="TensorBoard"
        )
        self.thread.daemon = True

    def start(self) -> None:
        self.thread.start()
        logger.info(
            f'Starting tensorboard on '
            f'{", ".join(f"http://{i}:{self.port}" for i in get_all_local_ip())}'
        )

    def shutdown(self) -> None:
        self.server.server_close()

    def join(self) -> None:
        self.thread.join()
