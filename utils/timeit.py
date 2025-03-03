import time

def timeit(should_time=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if should_time:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds to execute.")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator