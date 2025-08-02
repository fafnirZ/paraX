
# if we want to decorate the worker function
# make sure to decorate with normal functions not methods bound
# to a class, since that causes problems.

import functools

#
# this is needed so IPC can occur
# since pickle doesnt like nested functions
# so the most robust way is to define
# a top level class which decorates the 
# function
#
class ValidateKwargsOnly:
    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        if len(args) != 0:
            raise ValueError(f"Workers must be kwargs only, detected args: {args}")
        return self.func(**kwargs)



def apply_all_decorators(func):
    fn = ValidateKwargsOnly(func)
    return fn
