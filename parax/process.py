from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Optional
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
        tqdm_mode: Optional[Literal["normal", "multi"]] = None
    ):
        _worker_fn = (
            WorkerFunctionBuilder(worker_fn)
            .wrap(ValidateKwargsOnly)
            .wrap(ProcessAwareWorkerFunction)
            .get_function()
        )
        super().__init__(
            worker_fn=_worker_fn,
            worker_fn_kwargs=worker_fn_kwargs,
            num_workers=num_workers,
            batch_size=batch_size,
            tqdm_enabled=tqdm_enabled,
            tqdm_description=tqdm_description,
            tqdm_class=tqdm_class,
            tqdm_mode=tqdm_mode,
        )


    @property
    def executor_type(self) -> type[ProcessPoolExecutor]:
        return ProcessPoolExecutor