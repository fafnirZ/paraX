from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional
from parax.base import BaseExecutor
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from parax.decorators import ThreadAwareWorkerFunction, ValidateKwargsOnly, WorkerFunctionBuilder

if TYPE_CHECKING:
    from tqdm import tqdm

class ThreadedExecutor(BaseExecutor):

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
            .wrap(ThreadAwareWorkerFunction)
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
    
    @staticmethod
    def default_num_workers() -> int:
        """For threadpool, the default should be something large.
        
        Since, it is not CPU bound.
        """
        return 100 

    @property
    def executor_type(self) -> type[ThreadPoolExecutor]:
        return ThreadPoolExecutor