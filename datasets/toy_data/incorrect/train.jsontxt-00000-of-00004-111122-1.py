

def classic_connect(self):
    'Same as :func:`classic.connect <rpyc.utils.classic.connect>`, but with the ``host`` and \n        ``port`` parameters fixed'
    if (self.local_port is None):
        stream = SocketStream(self.remote_machine.connect_sock(stream.remote_port))
        return rpyc.classic.connect_stream(stream)
    else:
        return rpyc.classic.connect('localhost', self.local_port)
