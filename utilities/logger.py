'''Logging utility for Python applications.'''

import os
import sys
import logging

from logging.handlers import RotatingFileHandler


def activate_operations_logging(path='./', level=logging.DEBUG, log_size_mb=3):
    '''Configures logger to log all messages to a rotating file.
    
    Args:
        path (str, optional): Path to the log directory. Defaults to './', i.e.
            the current working directory.
        level (int, optional): Logging level. Defaults to 20, i.e. logging.DEBUG.
            See logging.__init__ for level definitions.
        log_size_mb (int, optional): Maximum log file size in MB before
            truncation. Defaults to 3 MB.
    '''
    os.makedirs(path, exist_ok=True)  # Ensure the logging directory exists
    log_file = os.path.join(path, 'MAIN.log')  # Set log file path
    
    # Configure root logger to log only to file
    logging.basicConfig(
        level=level,
        format='%(levelname)s | %(asctime)s | %(filename)s | %(funcName)s | %(lineno)d | %(message)s',
        datefmt='%d-%m-%Y %H:%M:%S',
        encoding='utf-8',
        handlers=[
            RotatingFileHandler(log_file, maxBytes=log_size_mb*1024*1024, backupCount=0)
        ]
    )
    
    # Log uncaught exceptions but still print them to console
    def log_unhandled_exceptions(exc_type, exc_value, exc_traceback):
        logging.critical('Uncaught Exception', exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)  # Preserve default console behavior
    
    sys.excepthook = log_unhandled_exceptions


class ResultsLogger:

    def __init__(self, path: str):
        self.path = path

    def log(self, message: str, console: bool = True):
        '''Logs a message to the console and a log file.
        
        Args:
            message (str): Message to log.
            console (bool, optional): Whether to print the message to the console.
        '''
        if console:
            print(message)
        with open(self.path, 'a') as f:
            f.write(f'{message}\n')


def serialize_callable(obj):
    '''Custom JSON serializer for callables.'''

    if callable(obj):
        '''Return function name instead of serializing the object.'''
        return obj.__name__  

    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')
