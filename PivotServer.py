"""
Class for launching a server to translate using pivot language
Inherits the Server class base functionality
"""
import socket
import xmlrpc.client
import subprocess
import psutil

import utilities

from Server import Server

class PivotServer(Server):

    def translate_interactive(self, working_dir1, working_dir2):
        """
        Launches the mosesdecoder to allow for interactive pivoting decoding
        from the source language into the pivot language and on to the
        target language
        """
        assert utilities.dir_exists(working_dir1), "TestInteractiveError: {} not found".format(working_dir1)
        assert utilities.dir_exists(working_dir2), "TestInteractiveError: {} not found".format(working_dir2)

        temp_file1 = working_dir1 + "/interactive.out"
        temp_file2 = working_dir2 + "/interactive.out"

        port1, port2 = self._get_free_port(), self._get_free_port()
        prox1, prox2 = self._setup_proxy(port1), self._setup_proxy(port2)

        process1 = self._load_server(working_dir1, port1, temp_file1)
        process2  =self._load_server(working_dir2, port2, temp_file2)

        self._manage_connections(prox1, prox2)

        self._shut_server(process1)
        self._shut_server(process2)

    def _manage_connections(self, prox1, prox2):
        """
        Accepts user input, submits it to moses for the pivoting
        translation and displays the result to the user
        """
        print("Enter text to translate (type quit to exit)")
        while True:
            query = input(">> ").lower().strip()
            if query == "quit" or query == "q":
                return
            try:
                piv_result = self._make_translation_request(prox1, query)
                tar_result = self._make_translation_request(prox2, piv_result)
            except (ConnectionRefusedError, xmlrpc.client.Fault) as e:
                tar_result = ''

            print("Text: {}\tTranslation: {}\n".format(query, tar_result))

def main():
    config = utilities.config_file_reader()
    path_to_moses = utilities.safe_string(config.get("Environment Settings", "path_to_moses_decoder"))

    server = PivotServer(path_to_moses)
    server.translate_interactive("es-en.working", "en-fr.working")

if __name__ == '__main__':
    main()
