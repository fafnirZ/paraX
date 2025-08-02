
from parax.base import BaseExecutor
from tqdm import tqdm


def test_basic():
    class BasicExecutor(BaseExecutor):
        def execute(self):
            pass

    def worker(**kwargs):
        pass

    kwargs = [{"arg": 1}, {"arg": 2}]
    
    executor = BasicExecutor(
        worker_fn=worker,
        worker_fn_kwargs=kwargs,
    )

    assert executor.tqdm_enabled == False
    assert executor.tqdm_description == ""
    assert executor.tqdm_class == tqdm


def test_description_enabled_tqdm():
    class BasicExecutor(BaseExecutor):
        def execute(self):
            pass
    def worker(**kwargs):
        pass
    kwargs = [{"arg": 1}, {"arg": 2}]
    
    executor = BasicExecutor(
        worker_fn=worker,
        worker_fn_kwargs=kwargs,
        tqdm_description="Something"
    )

    assert executor.tqdm_enabled == True
    assert executor.tqdm_description == "Something"
    assert executor.tqdm_class == tqdm


def test_explicit_flags_take_priority():
    class BasicExecutor(BaseExecutor):
        def execute(self):
            pass

    class CustomTQDM(tqdm):
        pass
    
    def worker(**kwargs):
        pass

    kwargs = [{"arg": 1}, {"arg": 2}]
    
    executor = BasicExecutor(
        worker_fn=worker,
        worker_fn_kwargs=kwargs,
        tqdm_enabled=False,
        tqdm_description="Something",
        tqdm_class=CustomTQDM,
    )

    assert executor.tqdm_enabled == False
    assert executor.tqdm_description == "Something"
    assert executor.tqdm_class == CustomTQDM

