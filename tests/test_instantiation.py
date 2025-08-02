
from parax.base import BaseExecutor


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

    assert executor 