# paraX
python multi-thread/processing boilerplate library w/ tqdm progress bar

The goal of this library is to provide a convenient interface for quickly spawning concurrent/parallel workflows
and to provide progress visualisation via tqdm, 
as well as to handle common issues such as:
- cancelling subsequent workers when one worker fails
- repeating it (if the worker is re-entrant)
- or skipping it.

## goals
- [x] ThreadPoolExecutor
- [ ] ProcessPoolExecutor
- [x] 1 single tqdm bar
- [ ] 1 bar per worker
- [x] cancel subsequent workers on failure
- [ ] retry failed workers
- [ ] skipping failed workers
- [ ] document performance issues with ProcessPoolExecutor re: serialisation of objects and provide alternative i.e. SHM use ZeroCopy frameworks.

## READTHIS
this section gives a good idea as to when ThreadPoolExecutors / ProcessPoolExecutors should be used.

this package is built completely on top of that, with the addition of useful boilerplate capabilities, like a progress bar and automatic cancellation of futures.

https://superfastpython.com/threadpoolexecutor-vs-processpoolexecutor/#Differences_Between_ThreadPoolExecutor_and_ProcessPoolExecutor
