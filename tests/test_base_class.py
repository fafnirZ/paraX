
import pytest
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


def get_kwargs():
    kwargs = [
        {"arg": 1}, 
        {"arg": 2},
        {"arg": 3}
    ]
    return kwargs

@pytest.mark.parametrize(
    "input, batch_size, expected_num_batches",
    [
        (get_kwargs(), 2, 2),
        ([*get_kwargs(), *get_kwargs()], 4, 2),
        (get_kwargs(), 100, 1),
    ]
)
def test_yield_batch(
    input, batch_size, expected_num_batches
):
    class BasicExecutor(BaseExecutor):
        def execute(self):
            pass

    class CustomTQDM(tqdm):
        pass
    
    def worker(**kwargs):
        pass

    
    executor = BasicExecutor(
        worker_fn=worker,
        worker_fn_kwargs=input,
        batch_size=batch_size
    )

    kwgs=[]
    num_batches=0
    for batch in executor.yield_batch():
        kwgs.extend(batch)
        num_batches+=1

    assert kwgs == input
    assert num_batches == expected_num_batches