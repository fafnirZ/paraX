
# if we want to decorate the worker function
# make sure to decorate with normal functions not methods bound
# to a class, since that causes problems.

from collections.abc import Callable
from dataclasses import dataclass
import functools
from typing import Any

#
# this is needed so IPC can occur
# since pickle doesnt like nested functions
# so the most robust way is to define
# a top level class which decorates the 
# function
#
class ValidateKwargsOnly:
    def __init__(self, func: Callable):
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        if len(args) != 0:
            raise ValueError(f"Workers must be kwargs only, detected args: {args}")
        return self.func(**kwargs)

@dataclass
class WorkerIdAndResultPacket:
    worker_id: int
    result: Any

class ThreadAwareWorkerFunction:
    def __init__(self, func: Callable):
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> tuple[int, Any]:
        import threading
        thread_id = threading.current_thread().ident
        results = self.func(**kwargs)
        return WorkerIdAndResultPacket(worker_id=thread_id, result=results)

class ProcessAwareWorkerFunction:
    def __init__(self, func: Callable):
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> tuple[int, Any]:
        import os
        process_id = os.getpid()
        results = self.func(**kwargs)
        return WorkerIdAndResultPacket(worker_id=process_id, result=results)



class WorkerFunctionBuilder:
    def __init__(self, func: Callable):
        self.func = func

    def wrap(self, decorator: Callable):
        self.func = decorator(self.func) 
        return self

    def get_function(self) -> Callable:
        return self.func


# I need a globally accessible function
# as a worker, such that processPool doesnt break
# i need this so I can wrap this in decorators which does stuff
def NOOP_function():
    pass

def NOOP_sleep_function():
    import time
    time.sleep(0.5)