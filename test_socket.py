# server.py
import asyncio
from io import BytesIO
import socket
import json
from typing import Any, Union

class Logger:
    prefix: str
    def print(self, *args):
        print(self.prefix, *args)

class ClusterSocketsController(Logger):
    addr: tuple[str, int]
    skt: socket.socket
    prefix = '[SERVER]:'

    def __init__(self, addr: tuple[str, int]):
        self.addr = addr
        self.skt = socket.create_server(self.addr)

    async def listen(self):
        self.skt.listen(1)
        conn, addr = self.skt.accept()
        coroutine = self.accept_connection(conn, addr)
        task = asyncio.create_task(coroutine)
        task.add_done_callback(lambda task: print('close'))

    async def accept_connection(self, conn: socket.socket, address: tuple[str, int]):
        host, port = address
        with BytesIO() as buffer:
            with conn:
                while True:
                    data = conn.recv(1024, )
                    buffer.write(data)
                    buffer.seek(0)
                    start_index = 0
                    for line in buffer:
                        start_index += len(line)
                        try:
                            parsed_msg: dict = json.loads(line)
                            self.on_experiment_update(parsed_msg)
                        except json.JSONDecodeError:
                            self.print(f'Error decoding msg "{line}" from {host}:{port}. Continuing anyway')
                    if start_index:
                        buffer.seek(start_index)
                        remaining = buffer.read()
                        buffer.truncate(0)
                        buffer.seek(0)
                        buffer.write(remaining)
                    else: buffer.seek(0, 2)


    def on_experiment_update(self, parsed_msg: dict):
        raise NotImplementedError()

class SocketsLogger(ClusterSocketsController):
    addr = ('', 8889)
    def __init__(self):
        super().__init__(self.addr)

    def on_experiment_update(self, update: Any):
        self.print(update)

# client.py
import socket
import json

class ClientSocket():
    def __init__(self, addr: tuple[str, int]):
        self.skt = socket.create_connection(addr)

    def send_metrics(self, metrics: dict[str, Union[float, str]]):
        self.skt.send(json.dumps(metrics).encode() + b'\n')

    def terminate(self):
        self.skt.close()
        

# main.py
async def main():
    ctl = SocketsLogger()
    client1 = ClientSocket(('localhost', ctl.addr[1]))
    client1.send_metrics({'train_loss': 3.2, 'experiment_id': 'asdhaus8d'})
    client1.send_metrics({'train_loss': 3.2, 'experiment_id': 'asdasd'})
    client1.send_metrics({'train_loss': 3.2, 'experiment_id': 'asdasd'})
    client1.send_metrics({'train_loss': 3.2, 'experiment_id': 'asdasd'})
    client1.terminate()
    print('client 1 says bye')
    await ctl.listen()


asyncio.run(main())
