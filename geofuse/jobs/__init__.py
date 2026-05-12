"""Worker callables that run inside the JobExecutor thread pool.

Modules in this package must not import ``streamlit`` — they receive every
piece of state they need (datasets, caches, engine instances) as plain
arguments from the executor.
"""
