import logging
from time import perf_counter



def count_time(func):
    def wrapper(*args, **kwargs):
        t_start = perf_counter()
        f = func(*args, **kwargs)
        t_end = perf_counter()
        print("elapsed time: ", t_end-t_start)
        return f
    return wrapper