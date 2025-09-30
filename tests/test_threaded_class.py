from parax.threading import ThreadedExecutor
import time

def test_threaded_executor():
    def worker(a, b):
        r = a**b
        time.sleep(0.2)
        return r

    tasks = [
        {"a": 10, "b": 2},
        {"a": 10, "b": 2},
        {"a": 10, "b": 2},
        {"a": 10, "b": 2},
        {"a": 10, "b": 2},
        {"a": 10, "b": 2},
    ] 

    exec = ThreadedExecutor(
        worker_fn=worker,
        worker_fn_kwargs=tasks,
        num_workers=2,
        tqdm_description="testing"
    )

    results = (
        exec
        .execute()
        .get_results()
    )

    assert all([
        res.result == 100
        for res in results
    ])

