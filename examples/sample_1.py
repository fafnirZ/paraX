"""
This example show:
for CPU intensive calculations,
Threadpool doesnt actually help because of GIL
"""

def calculate_pi_monte_carlo(*, iterations):
    """
    Calculates an approximation of Pi using the Monte Carlo method.
    This is a CPU-intensive task due to the large number of random
    number generations and mathematical operations.
    """
    import random

    points_inside_circle = 0
    for _ in range(iterations):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        distance = x**2 + y**2
        if distance <= 1:
            points_inside_circle += 1

    pi_approximation = 4 * points_inside_circle / iterations
    return pi_approximation



if __name__ == "__main__":
    print("[NOTE] use htop to monitor cpu usage.")
    from parax import ThreadedExecutor, ProcessExecutor
    import random
    requests = [
        {"iterations": random.randrange(10000,2000000)}
        for _ in range(25)
    ]

    print("[*] thread pool executor for cpu intensive") 
    (
        ThreadedExecutor(
            worker_fn=calculate_pi_monte_carlo,
            worker_fn_kwargs=requests,
            tqdm_description="calculating pi monte carlos [cpu bound] using threadpool..."
        )
        .execute()
    )

    print("[*] process pool executor for cpu intensive") 
    (
        ProcessExecutor(
            worker_fn=calculate_pi_monte_carlo,
            worker_fn_kwargs=requests,
            tqdm_description="calculating pi monte carlos [cpu bound] using threadpool..."
        )
        .execute()
    )