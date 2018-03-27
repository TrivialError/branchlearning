import _thread
import time
import math


def f(name):
    time.sleep(2)
    print('hello', name)


if __name__ == '__main__':
    for i in range(10):
        _thread.start_new_thread(f, ('bob',))
    time.sleep(10000000000000)
