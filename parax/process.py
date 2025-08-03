from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional
from parax.base import BaseExecutor
from concurrent.futures import Future, ProcessPoolExecutor, as_completed

from parax.decorators import ProcessAwareWorkerFunction, ValidateKwargsOnly, WorkerFunctionBuilder

if TYPE_CHECKING:
    from tqdm import tqdm

class ProcessExecutor(BaseExecutor):

    def __init__(
        self,
        *,
        worker_fn: Callable,
        worker_fn_kwargs: list[dict[str, Any]],
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,

        # tqdm related
        tqdm_enabled: Optional[bool] = None,
        tqdm_description: Optional[str] = None,
        tqdm_class: Optional[type[tqdm]] = None,
    ):
        _worker_fn = (
            WorkerFunctionBuilder(worker_fn)
            .wrap(ValidateKwargsOnly)
            .wrap(ProcessAwareWorkerFunction)
        )
        super().__init__(
            worker_fn=_worker_fn,
            worker_fn_kwargs=worker_fn_kwargs,
            num_workers=num_workers,
            batch_size=batch_size,
            tqdm_enabled=tqdm_enabled,
            tqdm_description=tqdm_description,
            tqdm_class=tqdm_class,
        )

    def execute(self) -> ProcessExecutor:
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            self._tqdm_init()
            for batch in self.yield_batch():
                completed_futures: set[Future] = set()
                all_futures: list[Future] = [
                    executor.submit(
                        self.worker_fn,
                        **kwargs_dict,
                    )
                    for kwargs_dict in batch
                ]

                # blocking until all futures in batch complete.
                for future in as_completed(all_futures):
                    self.handle_future(
                        future=future, 
                        all_futures=all_futures, 
                        completed_futures_mut=completed_futures
                    )
                    self._tqdm_update(1)

            self._tqdm_close()

        return self
