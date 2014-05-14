import os.path as op
import time
import shutil
from itertools import repeat

import numpy as np
from joblib import Memory
from IPython import __version__ as IPyversion
from IPython.parallel import Client

from .externals.six import print_


def _purge_results(client, task):
    """Purge results working around some IPython bugs"""
    if (IPyversion > '1.1.0' or IPyversion >= '0.11' and not client.outstanding):
        client.purge_results(task)


class ParallelCache(object):
    def __init__(self, cluster_profile=None, cachedir=None, purge=False,
                 idle_timeout=None, shutdown=False, retries=3, poll_interval=10,
                 verbose=5, **kwargs):
        self._purge = purge
        self._idle_timeout = idle_timeout
        self._shutdown = shutdown
        self._retries = retries
        self._poll_interval = poll_interval
        self._verbose = verbose
        self._execution_times = None
        if cluster_profile is not None:
            self._ip_client = Client(profile=cluster_profile, **kwargs)
        else:
            self._ip_client = None

        if cachedir is not None:
            self._memory = Memory(cachedir=cachedir, verbose=0)
        else:
            self._memory = None

    def map(self, f, *sequences, **kwargs):
        # make sure all sequences have the same length
        n_jobs = None
        my_seqs = []
        for ii, seq in enumerate(sequences):
            try:
                this_n_elems = len(seq)
                if n_jobs is None:
                    n_jobs = this_n_elems
                if this_n_elems != n_jobs:
                    raise ValueError('All sequences must have the same lenght,'
                                     'sequence at position %d has length %d'
                                     % (ii + 1, this_n_elems))
                my_seqs.append(seq)
            except TypeError:
                # we allow passing ints etc, convert them to a sequence
                my_seqs.append(repeat(seq))

        t_start = time.time()
        if self._ip_client is None:
            if self._verbose >= 1:
                tmp = 'without' if self._memory is None else 'with'
                print_('Running %d jobs locally %s caching..' % (n_jobs, tmp))
            out = list()
            my_fun = f if self._memory is None else self._memory.cache(f)
            for this_args in zip(*my_seqs):
                out.append(my_fun(*this_args, **kwargs))
        elif self._ip_client is not None and self._memory is None:
            if self._verbose >= 1:
                print('Running %d jobs on cluster without caching..' % n_jobs)
            out = [None] * n_jobs
            lbview = self._ip_client.load_balanced_view()
            tasks = list()
            for this_args in zip(*my_seqs):
                tasks.append(lbview.apply(f, *this_args, **kwargs))
            # wait for tasks to complete
            result_retrieved = [False] * len(tasks)
            execution_times = [None] * len(tasks)
            retry_no = np.zeros(len(tasks), dtype=np.int)
            last_print = 0
            last_idle_check = time.time()
            idle_times = {}
            while True:
                for ii, task in enumerate(tasks):
                    if not result_retrieved[ii] and task.ready():
                        if task.successful():
                            out[ii] = task.get()
                            execution_times[ii] = task.serial_time
                            result_retrieved[ii] = True
                        else:
                            # task failed for some reason, re-run it
                            if retry_no[ii] < self._retries:
                                if self._verbose > 3:
                                    print ('\nTask %d failed, re-running (%d / %d)'
                                           % (ii, retry_no[ii] + 1,
                                              self._retries))
                                this_args = zip(*my_seqs)[ii]
                                new_task = lbview.apply(f, *this_args, **kwargs)
                                tasks[ii] = new_task
                                retry_no[ii] += 1
                            else:
                                msg = ('\nTask %d failed %d times. Stopping'
                                       % (ii, self._retries + 1))
                                print msg
                                # this will throw an exception
                                task.get()
                                raise RuntimeError(msg)
                        if self._purge:
                            _purge_results(self._ip_client, task)

                n_completed = int(np.sum(result_retrieved))
                progress = n_completed / float(n_jobs)
                # print progress in 10% increments
                this_print = int(np.floor(progress * 10))
                if self._verbose >= 1 and this_print != last_print:
                    print_(' %d%%' % (100 * progress), end='')
                    last_print = this_print
                if n_completed == n_jobs:
                    # we are done!
                    print_('')  # newline
                    break
                if self._idle_timeout is not None and time.time() > last_idle_check + 30:
                    now = time.time()
                    queue = self._ip_client.queue_status()
                    shutdown_eids = []
                    for eid in self._ip_client.ids:
                        if eid not in queue:
                            continue
                        if queue[eid]['queue'] + queue[eid]['tasks'] == 0:
                            # engine is idle
                            idle_time = idle_times.get(eid, None)
                            if idle_time is None:
                                # mark engine as idle
                                idle_times[eid] = now
                                continue
                            if now - idle_time > self._idle_timeout:
                                # shut down engine
                                shutdown_eids.append(eid)
                        elif eid in idle_times:
                            # engine has started running again
                            del idle_times[eid]

                    if len(shutdown_eids) > 0:
                        if self._verbose > 0:
                            print 'Shuting-down engines: ', shutdown_eids
                        dv = self._ip_client.direct_view(shutdown_eids)
                        dv.shutdown()
                        for eid in shutdown_eids:
                            del idle_times[eid]
                    last_idle_check = now
                time.sleep(self._poll_interval)

            self._execution_times = execution_times
            if self._shutdown:
                self._shutdown_cluster()

        elif self._ip_client is not None and self._memory is not None:
            # now this is the interesting case..
            if self._verbose >= 1:
                print('Running %d jobs on cluster with caching..' % n_jobs)
            f_cache = self._memory.cache(f)
            lbview = None
            out = [None] * n_jobs
            execution_times = [None] * n_jobs
            task_info = list()

            n_cache = 0
            for ii, this_args in enumerate(zip(*my_seqs)):
                # get the cache directory
                out_dir, _ = f_cache.get_output_dir(*this_args, **kwargs)
                if op.exists(op.join(out_dir, 'output.pkl')):
                    out[ii] = f_cache.load_output(out_dir)
                    n_cache += 1
                    continue
                if lbview is None:
                    lbview = self._ip_client.load_balanced_view()
                task = lbview.apply(f, *this_args, **kwargs)
                task_info.append(dict(task=task, idx=ii, args=this_args))
            if self._verbose >= 1:
                print_('Loaded %d results from cache' % n_cache)

            # wait for tasks to complete
            last_print = 0
            last_idle_check = time.time()
            idle_times = {}
            result_retrieved = [False] * len(task_info)
            retry_no = np.zeros(len(task_info), dtype=np.int)
            failed_tasks = []
            while len(task_info) > 0:
                for ii, ti in enumerate(task_info):
                    if not result_retrieved[ii] and ti['task'].ready():
                        task = ti['task']
                        if task.successful():
                            this_out = task.get()
                            # cache the input and output
                            out_dir, _ = f_cache.get_output_dir(*ti['args'],
                                                                **kwargs)
                            f_cache._persist_output(this_out, out_dir)
                            f_cache._persist_input(out_dir, *ti['args'], **kwargs)
                            # insert result into output
                            out[ti['idx']] = this_out
                            execution_times[ti['idx']] = task.serial_time
                            result_retrieved[ii] = True
                        else:
                            if retry_no[ii] < self._retries:
                                if self._verbose > 3:
                                    print ('\nTask %d failed, re-running (%d / %d)'
                                           % (ii, retry_no[ii] + 1,
                                              self._retries))
                                new_task = lbview.apply(f, *ti['args'], **kwargs)
                                ti['task'] = new_task
                                retry_no[ii] += 1
                            else:
                                # task failed too many times, mark it as done
                                # but keep running
                                if self._verbose >= 1:
                                    print ('\nTask %d failed %d times.'
                                           % (ii, self._retries + 1))
                                failed_tasks.append(task)
                                result_retrieved[ii] = True

                    if self._purge:
                        _purge_results(self._ip_client, task)

                if self._idle_timeout is not None and time.time() > last_idle_check + 30:
                    now = time.time()
                    queue = self._ip_client.queue_status()
                    shutdown_eids = []
                    for eid in self._ip_client.ids:
                        if eid not in queue:
                            continue
                        if queue[eid]['queue'] + queue[eid]['tasks'] == 0:
                            # engine is idle
                            idle_time = idle_times.get(eid, None)
                            if idle_time is None:
                                # mark engine as idle
                                idle_times[eid] = now
                                continue
                            if now - idle_time > self._idle_timeout:
                                # shut down engine
                                shutdown_eids.append(eid)
                        elif eid in idle_times:
                            # engine has started running again
                            del idle_times[eid]

                    if len(shutdown_eids) > 0:
                        if self._verbose > 0:
                            print 'Shuting-down engines: ', shutdown_eids
                        dv = self._ip_client.direct_view(shutdown_eids)
                        dv.shutdown()
                        for eid in shutdown_eids:
                            del idle_times[eid]

                        last_idle_check = now

                n_completed = int(np.sum(result_retrieved))
                progress = n_completed / float(n_jobs - n_cache)
                # print progress in 10% increments
                this_print = int(np.floor(progress * 10))
                if self._verbose >= 1 and this_print != last_print:
                    print_(' %d%% ' % (100 * progress), end='')
                    last_print = this_print
                if n_completed == n_jobs - n_cache:
                    # we are done!
                    print_('')  # newline
                    break
                time.sleep(self._poll_interval)

            if self._shutdown:
                self._shutdown_cluster()

            if len(failed_tasks) > 0:
                msg = ''
                for task in failed_tasks[:5]:
                    try:
                        task.get()
                    except Exception as e:
                        msg += str(e)
                raise RuntimeError('%d tasks failed:\n %s'
                                   % (len(failed_tasks), msg))

            self._execution_times = execution_times
        else:
            raise RuntimeError('WTF?')

        if self._verbose >= 1:
            print_('Done (%0.1f seconds)' % (time.time() - t_start))

        return out

    def get_last_excecution_times(self):
        return self._execution_times

    def purge_results(self, f, *sequences, **kwargs):
        # make sure all sequences have the same length
        n_jobs = None
        my_seqs = []
        for ii, seq in enumerate(sequences):
            try:
                this_n_elems = len(seq)
                if n_jobs is None:
                    n_jobs = this_n_elems
                if this_n_elems != n_jobs:
                    raise ValueError('All sequences must have the same lenght,'
                                     'sequence at position %d has length %d'
                                     % (ii + 1, this_n_elems))
                my_seqs.append(seq)
            except TypeError:
                # we allow passing ints etc, convert them to a sequence
                my_seqs.append(repeat(seq))

        f_cache = self._memory.cache(f)
        n_deleted = 0
        for this_args in zip(*my_seqs):
            out_dir, _ = f_cache.get_output_dir(*this_args, **kwargs)
            if op.exists(out_dir):
                shutil.rmtree(out_dir)
                n_deleted += 1
        print 'Purging cache: %d out of %d deleted' % (n_deleted, n_jobs)

    def _shutdown_cluster(self):
        # shut down all idle engines
        queue = self._ip_client.queue_status()
        shutdown_eids = []
        for eid in self._ip_client.ids:
            if eid not in queue:
                continue
            if queue[eid]['queue'] + queue[eid]['tasks'] == 0:
                shutdown_eids.append(eid)
        if len(shutdown_eids) > 0:
            if self._verbose > 0:
                print 'Shuting-down engines: ', shutdown_eids
            dv = self._ip_client.direct_view(shutdown_eids)
            dv.shutdown()
