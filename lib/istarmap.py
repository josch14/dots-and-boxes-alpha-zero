import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """
    starmap-version of imap
    When using multiple threads to perform a loop of the same calculation, tqdm is not able to display actual progress
    correctly when starmap is used. Therefore, implement the patch istarmap() based on the code for imap().
    see: https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put((
        self._guarded_task_generation(result._job, mpp.starmapstar, task_batches), result._set_length
    ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap
