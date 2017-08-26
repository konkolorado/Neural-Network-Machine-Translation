"""
Base class for launching translation server
"""
import socket
import xmlrpc.client
import subprocess
import psutil

import utilities

class Server(object):
    def __init__(self, path_to_moses, verbose=False):
        self.path_to_moses = path_to_moses
        self.verbose = verbose

    def _print(self, item):
        if self.verbose:
            utilities.flush_print(item)

    def translate_interactive(self, working_dir):
        """
        Launches the mosesdecoder to allow for interactive decoding between
        source and target languages.
        """
        assert utilities.dir_exists(working_dir), "TestInteractiveError: {} not found".format(working_dir)

        temp_file = working_dir + "/interactive.out"
        port = self._get_free_port()
        proxy = self._setup_proxy(port)

        process = self._load_server(working_dir, port, temp_file)
        self._manage_connections(process, proxy)
        self._shut_server(process)

    def _manage_connections(self, process, proxy):
        """ Accepts user input, submits it to moses, returns the result """
        print("Enter text to translate (type quit to exit)")
        while True:
            query = input(">> ").lower().strip()
            if query == "quit" or query == "q":
                return

            try:
                result = self._make_translation_request(proxy, query)
            except (ConnectionRefusedError, xmlrpc.client.Fault) as e:
                result = ''

            print("Text: {}\tTranslation: {}\n".format(query, result))

    def _load_server(self, working_dir, port, logfile):
        """
        Loads the moses server on the provided port number using the
        informatio in the specified working directory. Returns the
        launched server process
        """
        self._print("Loading interactive translator at {}...".format(working_dir))
        command = self.path_to_moses + "bin/moses" + \
            " -minlexr-memory --server --server-port {}".format(port) + \
            " --server-maxconn-backlog 5" + \
            " -v 0 -f {}/mert-work/moses.ini &".format(working_dir)

        with open(logfile, 'w') as err:
            process = subprocess.Popen(command.split(), shell=False, stderr=err)
        self._print("Ready\n")
        return process

    def _make_translation_request(self, proxy, text):
        """ Sends the text we want to translate to the moses server """
        response = proxy.translate({"text": text})
        return response["text"]

    def _get_free_port(self):
        """ Returns an available port number for moses server to use """
        sock = socket.socket()
        sock.bind(('', 0))
        return sock.getsockname()[1]

    def _setup_proxy(self, port):
        """ Sets up a server proxy to communicate with moses server """
        return xmlrpc.client.ServerProxy("http://localhost:{}/RPC2".format(port))

    def _shut_server(self, process):
        """ Uses the putil library to shut the background translation
        service down """
        psutil.Process(process.pid).kill()

def main():
    config = utilities.config_file_reader()
    path_to_moses = utilities.safe_string(config.get("Environment Settings", "path_to_moses_decoder"))

    server = Server(path_to_moses)
    server.translate_interactive("es-en.working")

if __name__ == '__main__':
    main()
