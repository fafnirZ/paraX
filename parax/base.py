from __future__ import annotations
from abc import ABC, abstractmethod
from concurrent.futures import Future, as_completed
from typing import TYPE_CHECKING, Any, Generator, Literal, Optional, Type, overload
from collections.abc import Callable


if TYPE_CHECKING:
    from tqdm import tqdm
    from concurrent.futures import Executor

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
    tqdm_mode: Literal["normal", "multi"]

    @property
    @abstractmethod
    def executor_type(cls) -> type[Executor]:
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
        tqdm_mode: Optional[Literal["normal", "multi"]] = None
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
        self.tqdm_mode = tqdm_mode or self.default_tqdm_mode()

        self.validate_attributes()
        self.results = []

    ############################
    # initialisation functions #
    ############################

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
    def default_tqdm_mode() -> str:
        return "normal"

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
            match self.tqdm_mode:
                case "normal":
                    self.tqdm_instance = self.tqdm_class(
                        total=len(self.worker_fn_kwargs),
                        desc=self.tqdm_description,
                    )
                case "multi":
                    if self.num_workers > 100:
                        print("[WARNING]: its getting to the point of this not being visually useful...")
                    self.tqdm_instances = {
                        index: self.tqdm_class(
                            position=index,
                            total=int(len(self.worker_fn_kwargs)/self.num_workers), # initialise as evenly distributed, but will update total dynamically later.
                            desc=self.tqdm_description,
                        )
                        for index in range(self.num_workers)
                    }
                case _ :
                    raise ValueError(f"Invalid tqdm mode, expected 'normal' or 'multi', instead got {self.tqdm_mode}")

    def _tqdm_update(self, **kwargs):
        if self._is_tqdm_enabled():
            if not self.tqdm_instance:
                raise RuntimeError("Missing tqdm instance for some reason, did you call _tqdm_init()?")
            
            match self.tqdm_mode:
                case "normal":
                    amount = kwargs.get("amount", None)
                    if not amount:
                        raise RuntimeError("tqdm update failure.")
                    self.tqdm_instance.update(amount) 
                case "multi":
                    if len(kwargs) != 2:
                        raise RuntimeError(f"Expected inputs to this function to be 2, got {len(kwargs)}")
                    amount = kwargs.get("amount", None)
                    worker_id = kwargs.get("worker_id", None)
                    if not amount:
                        raise RuntimeError("tqdm update failure, 'amount' is None") 
                    if not worker_id:
                        raise RuntimeError("tqdm update failure, 'worker_id' is None")
        # else NOOP

    @overload
    def _tqdm_update(self, *, amount:int):
        pass
    @overload
    def _tqdm_update(self, *, amount:int, worker_id: int):
        pass

    def _init_workers_tqdm_map(self, executor: Executor):
        """This function will perform the following.
        
        Initialise 1 thread/process for each worker.
        Get the process/thread_id, and return it, we then create
        an index -> process/thread_id map, such that we can 
        map the process/thread_id to an index for it to update a
        tqdm instance at that index.
        """
        pass 

    def _tqdm_close(self):
        if self._is_tqdm_enabled():
            if not self.tqdm_instance:
                raise RuntimeError("Missing tqdm instance for some reason, did you call _tqdm_init()?")
            self.tqdm_instance.close()
        # else NOOP

    
    def get_results(self) -> list[Any]:
        return self.results

    def execute(self) -> BaseExecutor:
        with self.executor_type(max_workers=self.num_workers) as executor:
            self._tqdm_init()
            self._init_workers_tqdm_map(executor)
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
    
    ###################
    # handles futures #
    ###################
    def handle_future(
        self,
        *, 
        future: Future,
        all_futures: list[Future],
        completed_futures_mut: set[Future],
    ):
        try:
            result = future.result()
            self.results.append(result)
            completed_futures_mut.add(future)

        except Exception as e:
            # cancels all incomplete futures on any exception being raised
            for future in self.tqdm_class(all_futures, desc="Cancelling incomplete futures"):
                if self.has_this_future_already_completed(future, completed_futures_mut):
                    continue
                # cancel incomplete
                future.cancel()
            print("Exeption encountered, cancelled all incomplete futures.")
            raise e
                
    
    @staticmethod
    def has_this_future_already_completed(future: Future, completed_futures: set[Future]) -> bool:
        if (future in completed_futures) and (future.done()):
            return True
        return False

        

