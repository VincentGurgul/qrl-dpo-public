'''Module containing function wrappers.'''

import os
import sys
import time
import logging
import numpy as np

from functools import lru_cache, wraps
from humanfriendly import format_timespan

from utilities.telegram import send_message


def timeit(func):
    '''Prints time that it took a function to run after successful execution.'''

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"Function {func.__name__} took {format_timespan(duration)} to run.")
        return result
    
    return wrapper

    
def retry(max_tries: int = 3, delay_seconds: int = 1):
    '''Tries to run a function 'max_tries' times with a 'delay_seconds' delay
    between runs.
    
    Args:
        max_tries (int, optional): Maximum number of tries. Defaults to 3.
        delay_seconds (int, optional): Delay between tries. Defaults to 1.
    '''
    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            tries = 0
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    print(f'(Retry wrapper) Function {func.__name__} raised error during execution no. {tries}: {type(e).__name__} ({e})')
                    if tries == max_tries:
                        raise e
                    time.sleep(delay_seconds)

        return wrapper
    
    return decorator


def log_execution(level=20, path='./'):
    '''Decorator factory that sets up a logger and logs function execution.

    Args:
        level (int, optional): Logging level. Defaults to 20, i.e. logging.DEBUG.
            See logging.__init__ for level definitions.
        path (str, optional): Path to the log file. Defaults to './', i.e. the
            current working directory.
    '''
    os.makedirs(path, exist_ok=True)  # Ensure the logging directory exists

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            '''Wrapper function that logs the execution of the decorated
            function.
            '''
            # Configure logging
            logging.basicConfig(
                filename=os.path.join(path, 'MAIN.log'),
                format='%(levelname)s | %(asctime)s | %(message)s',
                datefmt='%d.%m.%Y %H:%M:%S',
                encoding='utf-8',
            )
            logging.log(level, f'Executing {func.__name__}')
            
            # Execute function and catch exceptions
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logging.error(f'Function {func.__name__} raised error during execution: {type(e).__name__} ({e})')
                raise e
                
            logging.log(level, f'Finished executing {func.__name__}')
            return result
        
        return wrapper
    
    return decorator


def progress_bar(
    iterable,
    length=None,
    bar_size=30,
    prefix='',
    suffix='',
    fill='█',
    print_end='\n',
):
    '''A custom progress bar function that overwrites itself in the terminal.

    Args:
        iterable (iterable): The iterable to loop over.
        length (int, optional): Length of the progress bar (in characters).
            Defaults to None (i.e. will be inferred from iterable).
        bar_size (int, optional): Length of the displayed bar. Defaults to 30 elements.
        prefix (str, optional): Prefix string to display before the progress bar.
        suffix (str, optional): Suffix string to display after the progress bar.
        fill (str, optional): Character to fill the progress bar with. Defaults to '█'.
        print_end (str, optional): End character (e.g., '\r' to overwrite,
            '\n' for new line). Defaults to '\n'.
        
    Yields:
        The elements from the provided iterable, one at a time.
    '''
    if length is None:
        length = len(iterable)

    def print_bar(progress, elapsed_time):
        percent = 100 * (progress / float(length))
        filled_length = int(bar_size * progress // length)
        bar = fill * filled_length + '-' * (bar_size - filled_length)
        sys.stdout.flush()
        sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix} [Elapsed: {format_timespan(elapsed_time)}]')

    print_bar(0, 0)

    start_time = time.time()

    for i, item in enumerate(iterable, 1):
        yield item
        elapsed = time.time() - start_time
        print_bar(i, elapsed)

    sys.stdout.write(print_end)


def np_cache(*args, **kwargs):
    '''LRU cache implementation for functions whose first parameter is a numpy
    array. The numpy array is converted to a hashable tuple before being passed
    to the function.
    '''
    def decorator(function):

        @wraps(function)
        def wrapper(np_array, *args, **kwargs):
            '''Wrapper function that converts the numpy array to a hashable
            tuple before passing it to the cached function.
            '''
            hashable_array = array_to_tuple(np_array)
            return cached_wrapper(hashable_array, *args, **kwargs)

        @lru_cache(*args, **kwargs)
        def cached_wrapper(hashable_array, *args, **kwargs):
            '''Cached wrapper function that takes a hashable tuple as input.'''
            array = np.array(hashable_array)
            return function(array, *args, **kwargs)

        def array_to_tuple(np_array):
            '''Converts a numpy array to a hashable tuple.'''
            try:
                return tuple(array_to_tuple(_) for _ in np_array)
            except TypeError:
                return np_array

        # Add cache info and clear methods to the wrapper
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear

        return wrapper

    return decorator


def telegram_notify(func):
    ''' Sends telegram message confirming successfull completion or informing
    of erroneous termination of function execution. '''
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
            send_message(f'Function {func.__name__} successfully executed \U0001F389')
            return output
        except Exception as e:
            send_message(f'Function {func.__name__} returned error: {e} \U0001F614')
            raise e
                
    return wrapper
