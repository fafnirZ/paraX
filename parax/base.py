from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generator, Optional, Self, Type
from collections.abc import Callable

if TYPE_CHECKING:
    from tqdm import tqdm

class BaseExecutor(ABC):
    """Base executor which outlines the general template which all executors follow.
    
    Abstract_methods:
        execute()
    
    Attributes:
        .worker_fn
        .worker_fn_kwargs
        .num_workers (default: user's cpu count)
        .batch_size (default: 1000)
        .results

        .tqdm_enabled
        .tqdm_description
        .tqdm_class
        .tqdm_instance

    Methods:
        ._is_tqdm_enabled
        ._tqdm_init
        ._tqdm_update
        ._tqdm_close
    """
    worker_fn: Callable
    worker_fn_kwargs: list[dict[str, Any]]
    num_workers: int
    batch_size: int
    results: list[Any]


    tqdm_enabled: bool
    tqdm_description: str
    tqdm_class: Type[tqdm] # must be a subclas of tqdm
    tqdm_instance: tqdm | None 

    @abstractmethod
    def execute(self) -> Self:
        raise NotImplementedError("The BaseClass should implement this.")

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
        self.worker_fn = worker_fn
        self.worker_fn_kwargs = worker_fn_kwargs
        self.num_workers = num_workers or self.default_num_workers()
        self.batch_size = batch_size or self.default_batch_size()
        
        (self.tqdm_enabled, self.tqdm_description, self.tqdm_class) = self.init_tqdm(
            input_tqdm_enabled=tqdm_enabled, 
            input_tqdm_description=tqdm_description, 
            input_tqdm_class=tqdm_class,
        )

        self.validate_attributes()
        self.results = []

    @staticmethod
    def init_tqdm(
        *,
        input_tqdm_enabled: Optional[bool],
        input_tqdm_description: Optional[str],
        input_tqdm_class: Optional[type[tqdm]],
    ) -> tuple[bool, str, type[tqdm]]:
        """tqdm is not enabled unless a description is provided.""" 
        if input_tqdm_enabled is False: # if user explicitly disables
            return (
                False,
                input_tqdm_description or "",
                input_tqdm_class or BaseExecutor.default_tqdm_class(),
            )
        else:
            if input_tqdm_description is not None:
                # we must enable tqdm if they provide a description
                # even if the user doesn't provide enabled
                _tqdm_enabled = True
                _tqdm_description = input_tqdm_description
            else:
                _tqdm_enabled = False
                _tqdm_description = ""

            _tqdm_class = input_tqdm_class or BaseExecutor.default_tqdm_class()
            return (
                _tqdm_enabled,
                _tqdm_description,
                _tqdm_class,
            )

    @staticmethod
    def default_tqdm_class() -> type[tqdm]:
        from tqdm import tqdm # tqdm_std
        return tqdm

    @staticmethod
    def default_num_workers() -> int:
        """Determine num workers from user's system."""
        import os
        return os.cpu_count()
    
    @staticmethod
    def default_batch_size() -> int:
        """Batch size explanation.

        Explanation:
            You want your batch size to be sufficiently large enough such that your workers are
            always pre-occupied, but at the end of processing a batch, there is a point in time
            where some workers are going to be idle.
            
            so if you set the batch size == num_workers then for each batch the bottleneck 
            would be the time it takes for the longest worker to complete its job.

            why batch then cant we just submit all jobs at once?
            well then you incur the initialisation costs of submitting all jobs at once,
            which may cause you to OOM.

            so batches is a mechanism for me to chunk the job into appropriate sizes such that
            I can handle huge jobs.

            the number of batches will equal to `math.ceil(len(worker_fn_kwargs)/batch_size)`
        """
        return 1000
    
    def validate_attributes(self):
        from tqdm import tqdm

        def generate_error_message(attribute_name: str, expected_type: type) -> str:
            attribute_value = getattr(self, attribute_name)
            attribute_value_type = type(attribute_value)
            error_msg = f"expected type {expected_type} for attribute self.{attribute_name}, instead got type {attribute_value_type}"
            return error_msg

        if not isinstance(self.worker_fn, Callable):
            msg = generate_error_message("worker_fn", Callable)
            raise TypeError(msg)

        for kwargs in self.worker_fn_kwargs:
            if not isinstance(kwargs, dict):
                raise TypeError(
                    "expected self.worker_fn_kwargs to be a list of dicts"
                    f"instead encountered: {type(kwargs)}"
                )

        if not isinstance(self.num_workers, int): 
            msg = generate_error_message("num_workers", int)
            raise TypeError(msg)

        if not isinstance(self.batch_size, int): 
            msg = generate_error_message("batch_size", int)
            raise TypeError(msg)
        
        if not isinstance(self.tqdm_enabled, bool): 
            msg = generate_error_message("tqdm_enabled", bool)
            raise TypeError(msg)

        if not isinstance(self.tqdm_description, str): 
            msg = generate_error_message("tqdm_description", str)
            raise TypeError(msg)

        if not issubclass(self.tqdm_class, tqdm): 
            msg = f"provided tqdm_class {self.tqdm_class} is not a subclass of tqdm"
            raise TypeError(msg)

    def yield_batch(self) -> Generator[list[dict], Any, Any]:
        for chunk_start_index in range(0, len(self.worker_fn_kwargs), self.batch_size):
            yield self.worker_fn_kwargs[chunk_start_index: chunk_start_index+self.batch_size]

    
    ########
    # tqdm #
    ########
    def _is_tqdm_enabled(self) -> bool:
        return self.tqdm_enabled

    def _tqdm_init(self) -> tqdm | None:
        if self._is_tqdm_enabled():
            self.tqdm_instance = self.tqdm_class(
                total=len(self.worker_fn_kwargs),
                desc=self.tqdm_description,
            )

    def _tqdm_update(self, amount: int):
        if self._is_tqdm_enabled():
            if not self.tqdm_instance:
                raise RuntimeError("Missing tqdm instance for some reason, did you call _tqdm_init()?")
            self.tqdm_instance.update(amount) 
        # else NOOP

    def _tqdm_close(self):
        if self._is_tqdm_enabled():
            if not self.tqdm_instance:
                raise RuntimeError("Missing tqdm instance for some reason, did you call _tqdm_init()?")
            self.tqdm_instance.close()
        # else NOOP

    
    def get_results(self) -> list[Any]:
        return self.results

   
