"""
This code is adapted from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

The code only enables colors in log messages, if the colorlog module is installed and if the output actually goes to a
terminal. This avoids escape sequences being written to a file when the log output is redirected.

Also, a custom color scheme is setup that is better suited for terminals with dark background.
"""

import logging
import os
import sys
try:
    import colorlog
except ImportError:
    pass

def setup_logging():
    terminal_logger = logging.getLogger()
    terminal_logger.propagate = False
    stderr_format = '%(asctime)s -  %(name)-10s - %(levelname)-8s - %(message)s'
    stdout_format = '%(asctime)s - %(levelname)-8s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    # ensuring that we are writing to a terminal and not a file
    # this avoids escape sequences being written to a file when the log output is redirected
    if 'colorlog' in sys.modules and os.isatty(2):
        cformat = '%(log_color)s' + stdout_format
        stdout_format = colorlog.ColoredFormatter(
            cformat,
            date_format,
            log_colors = {
                'DEBUG': 'reset',
                'INFO' : 'reset',
                'WARNING' : 'bold_yellow',
                'ERROR': 'bold_red',
                'CRITICAL': 'bold_red'
            }
        )
    else:
        stdout_format = logging.Formatter(stdout_format, date_format)

    # to avoid duplicate logging output as pytorch lightning defines its own python logger `python_logging`
    if (terminal_logger.hasHandlers()):
        terminal_logger.handlers.clear()

    # create file handler which logs even debug messages
    fh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(stdout_format)
    terminal_logger.addHandler(fh)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter(stderr_format))
    terminal_logger.addHandler(ch)

setup_logging()

def get_terminal_logger(name):
    return logging.getLogger(name)