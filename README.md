# paraX
python multi-thread/processing boilerplate library w/ tqdm progress bar

The goal of this library is to provide a convenient interface for quickly spawning concurrent/parallel workflows
and to provide progress visualisation via tqdm, 
as well as to handle common issues such as:
- cancelling subsequent workers when one worker fails
- repeating it (if the worker is re-entrant)
- or skipping it.

## goals
- [ ] ThreadPoolExecutor
- [ ] ProcessPoolExecutor
- [ ] 1 single tqdm bar
- [ ] 1 bar per worker
- [ ] cancel subsequent workers on failure
- [ ] retry failed workers
- [ ] skipping failed workers
- [ ] document performance issues with ProcessPoolExecutor re: serialisation of objects and provide alternative i.e. SHM use ZeroCopy frameworks.
