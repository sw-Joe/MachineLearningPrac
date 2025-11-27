import logging as log
import time



now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

log.basicConfig(
    filename=f'ml_{now}.log',
    level=log.INFO,
    format='%(asctime)s|%(levelname)s| %(message)s'
)