import torch


def pytest_sessionstart(session):
    # Tiny matrix operations are slower with a large CPU thread pool.
    torch.set_num_threads(1)
