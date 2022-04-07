#!/usr/bin/env python

from typing import Optional
import traceback
from multiprocessing import Queue, Process, Pipe, cpu_count
from queue import Empty as EmptyQueueError
import logging
import sys
import os
from time import sleep

import h5py

from deeprank_gnn.domain.amino_acid import amino_acids
from deeprank_gnn.models.query import SingleResidueVariantAtomicQuery


_log = logging.getLogger(__name__)


class _PreProcess(Process):
    def __init__(self, input_queue, output_path, error_queue):
        Process.__init__(self)
        self.daemon = True

        self._input_queue = input_queue
        self._error_queue = error_queue

        self._output_path = output_path

        self._receiver, self._sender = Pipe(duplex=False)

    @property
    def output_path(self):
        return self._output_path

    def is_running(self):
        with self._running_lock:
            return self._running

    def stop(self):
        self._sender.send(1)
        self.join()

    def _should_stop(self):
        return self._receiver.poll()

    def run(self):
        while not self._should_stop():
            try:
                query = self._input_queue.get_nowait()

            except EmptyQueueError:
                continue

            graph = None
            try:
                graph = query.build_graph()
                if graph.has_nan():
                    self._error_queue.put(f"skipping {query}, because of a generated NaN value in the graph")

                graph.write_to_hdf5(self._output_path)
            except:
                self._error_queue.put(traceback.format_exc())

                # Don't leave behind an unfinished hdf5 group.
                if graph is not None:
                    with h5py.File(self._output_path, 'a') as f5:
                        if graph.id in f5:
                            del f5[graph.id]


class PreProcessor:
    "preprocesses a series of graph building operations (represented by queries)"

    def __init__(self, prefix: Optional[str] = None, process_count: Optional[int] = None):
        """
        Args:
            prefix(str, optional): prefix for the output files, ./preprocessed-data- by default
            process_count(int, optional): how many subprocesses will I run simultaneously, by default takes all available cpu cores.
        """

        self._input_queue = Queue()
        self._error_queue = Queue()

        if process_count is None:
            process_count = cpu_count()

        if prefix is None:
            prefix = "preprocessed-data"

        self._processes = [_PreProcess(self._input_queue, "{}-{}.hdf5".format(prefix, index), self._error_queue)
                           for index in range(process_count)]

    def start(self):
        "start the workers"

        _log.info("starting {} worker processes".format(len(self._processes)))
        for process in self._processes:
            process.start()
            if not process.is_alive():
                raise RuntimeError(f"worker process {process.name} did not start")

    def wait(self):
        "wait for all graphs to be built"

        try:
            _log.info("waiting for the queue to be empty..")
            while self._input_queue.qsize() > 0:  # qsize seems to be more reliable than the empty method
                sleep(1.0)
        finally:
            self.shutdown()

        # log all errors in the subprocesses
        while not self._error_queue.empty():
            error_s = self._error_queue.get()
            _log.error(error_s)

    def add_query(self, query):
        "add a single graph building query"

        self._input_queue.put(query)

    def add_queries(self, queries):
        "add multiple graph building queries"

        for query in queries:
            self._input_queue.put(query)

    @property
    def output_paths(self):
        return [process.output_path for process in self._processes
                if os.path.isfile(process.output_path)]

    def shutdown(self):
        "stop building graphs"

        _log.info("shutting down {} worker processes..".format(len(self._processes)))

        for process in self._processes:
            process.stop()

    def __del__(self):
        _log.debug("del called on preprocessor")

        # clean up all the created subprocesses
        self.shutdown()

