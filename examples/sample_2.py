"""
This example show:
for many IO bound requests, 

Threadpool is better
Since you can num_threads many times greater than num_processes

whilst you can create an insane number of processes, its not recommended.
"""

def random_sleep(*, sleep_time):
    import time
    time.sleep(sleep_time)





if __name__ == "__main__":
    print("[NOTE] use htop to monitor cpu usage.")
    from parax import ThreadedExecutor, ProcessExecutor
    import random
    requests = [
        {"sleep_time": random.uniform(0.1, 2.0)}
        for _ in range(10000)
    ]

    print("[*] thread pool executor for IO Bound") 
    (
        ThreadedExecutor(
            worker_fn=random_sleep,
            worker_fn_kwargs=requests,
            num_workers=1000,
            tqdm_description="calculating sleep [IO bound] using threadpool..."
        )
        .execute()
    )

    print("[*] process pool executor for IO Bound") 
    (
        ProcessExecutor(
            worker_fn=random_sleep,
            worker_fn_kwargs=requests,
            num_workers=1000,
            tqdm_description="calculating sleep [IO bound] using threadpool..."
        )
        .execute()
    )

    print("[!] clear winner for Threapool here, it also doesnt use as much CPU for no reason...")


    