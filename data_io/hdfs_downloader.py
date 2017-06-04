
import os
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import multiprocessing as mp
import subprocess
#import logging
from tensorflow import logging
import tempfile


def fetch_hdfs_data(paths, data_msg_q, retry_times = 3, data_dir=None):
  msg = data_msg_q.get_msg()
  if data_dir and not os.path.isdir(data_dir):
    try:
      data_dir = tempfile.mkdtemp(prefix='data_', suffix='_tmp', dir='./')
    except Exception, e:
      logging.error(e)
      data_dir = './'

  while True:
    logging.debug("receive msg: " + msg)
    if msg == 'reset':
      for data_path in paths:
        filename = os.path.split(data_path)[1]
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
          logging.debug(filename + " all readly exist in local")
          data_msg_q.put_data(filepath)
          logging.debug("reuse local data " + filepath + " done")
          continue
        count, ret_code = 0, -1
        command = ["hadoop", "fs", "-get", data_path, data_dir]
        while count < retry_times and ret_code != 0:
          try:
            ret_code = subprocess.check_call(command)
          except subprocess.CalledProcessError, e:
            logging.error(e)
            count += 1

        if ret_code == 0:
          data_msg_q.put_data(filepath)
          logging.debug("fetch " + data_path + " done")
        else:
          logging.debug("fetch " + data_path + " failed")

      data_msg_q.put_data(None)
    elif msg == 'shuffle':
      random.shuffle(paths)
    elif msg == 'Done':
      break
    msg = data_msg_q.get_msg()

class DataMsgQueue(object):
  """
      two queue, one for data transfer, one for communication between processors
  """
  def __init__(self, max_data_queue_size):
    self.__data_q = mp.Queue(max_data_queue_size)
    self.__msg_q = mp.Queue()
    self.__closed = False
    self.__blocked = False

  def get_data(self, timeout=None):
    if self.__closed:
        return None
    try:
      return self.__data_q.get(timeout=timeout)
    except Exception:
      return None

  def get_msg(self, timeout=None):
    try:
      return self.__msg_q.get(timeout=timeout)
    except Exception:
      return None

  def __iter__(self):
    while True:
      data = self.__data_q.get()
      if data is None:
        break
      yield data

  def put_data(self, val):
    if not self.__closed and not self.__blocked:
        self.__data_q.put(val)

  def put_msg(self, val):
    self.__msg_q.put(val)

  def shuffle(self):
    self.__msg_q.put('shuffle')

  def remove(self, filename):
    if os.path.exists(filename):
      try:
        ret_code = subprocess.check_call(["rm", filename])
        logging.debug("delete local file " + filename)
      except subprocess.CalledProcessError, e:
        logging.error(e)

  def close(self):
    self.put_data(None)
    self.put_msg('Done')
    self.__blocked = True

    filename = self.get_data(timeout=2)
    while filename:
      self.remove(filename)
      filename = self.get_data(timout=2)
    logging.debug("clean data queue && close")
    self.__closed = True

class HDFSDownloader(object):
    def __init__(self, data_pattern, save_dir, capacity=100, min_cache=5, num_downloader=4):
        self.check_data_pattern(data_pattern)
        self.save_dir = save_dir
        self.capacity = capacity
        self.min_cache = min_cache
        self.init_out_queue()
        self._file_queue = DataMsgQueue(128)
        self._curr_files = []
        self.num_downloader = num_downloader
        self._processes = []
        self._downloading = False
        self._finished = 0

    def check_data_pattern(self, data_pattern):
        if not data_pattern.startswith('hdfs://haruna'):
            raise ValueError('data pattern is not hdfs files: data pattern should starts with \'hdfs://haruna\'')
        self.data_pattern = data_pattern.replace('hdfs://haruna', '')

        self.data_dir, self.data_pattern = os.path.split(self.data_pattern)
        # check if data_pattern exist
        command = ["hadoop", "fs", "-test", "-d", self.data_dir]
        ret_code = subprocess.check_call(command)
        if ret_code != 0:
            raise IOError('No such file in HDFS: dir {}, data pattern {}'.format(self.data_dir, self.data_pattern))
        self._get_file_list()

    def _get_file_list(self):
        command = ["hadoop", "fs", "-ls", os.path.join(self.data_dir, self.data_pattern)]
        self.file_list = subprocess.check_output(command)
        self.file_list = self.file_list.split('\n')
        self.file_list = [f.split(' ')[-1] for f in self.file_list[1:]]

    @property
    def num_files(self):
        return len(self.file_list)

    def init_out_queue(self):
        self.out_que = data_flow_ops.FIFOQueue(capacity=self.capacity,
                                               dtypes=tf.string,
                                               shapes=[],
                                               name='hdfs_queue')
        self._input_files = tf.placeholder(tf.string, shape=[None, ])
        self._enqueue_op = self.out_que.enqueue_many([self._input_files])
        self._queue_size_op = self.out_que.size()
        self._enqueued = 0

    def get_out_queue(self):
        return self.out_que

    def enqueue(self, sess):
        # first remove used local files
        #if len(self._curr_files) > 1:
        #    for f in self._curr_files:
        #        self._file_queue.remove(f)
        self._curr_files = []
        # enqueue
        if self._finished >= self.num_downloader or not self._downloading:
            return 1
        while len(self._curr_files) < self.min_cache:
            f = self._file_queue.get_data()
            if f is None:
                self._finished += 1
                continue
            logging.debug('Got file {}'.format(f))
            self._curr_files.append(f)
            if self._finished >= self.num_downloader or not self._downloading:
                break
        outputs = sess.run([self._enqueue_op, self._queue_size_op], feed_dict={self._input_files: self._curr_files})
        logging.debug('Output queue size: {}'.format(outputs[1]))
        self._enqueued += self.min_cache
        return 0

    def enqueuing(self, sess, num_epochs=1):
        ret_code = 0
        count = 0
        while ret_code == 0:
            ret_code = self.enqueue(sess)
            if ret_code != 0:
                count += 1
                if count < num_epochs:
                    self.reset()
                    ret_code = 0

    def start_downloading(self):
        if self._downloading:
            return
        num_downloader = min(self.num_downloader, self.file_list)
        self.num_downloader = num_downloader
        download_list = [[] for _ in range(num_downloader)]
        for idx, f in enumerate(self.file_list):
            which_idx = idx % num_downloader
            download_list[which_idx].append(f)
        self._processes = []
        for idx in range(num_downloader):
            p = mp.Process(target=fetch_hdfs_data,
                    args=(download_list[idx], self._file_queue),
                    kwargs={'data_dir': self.save_dir})
            p.daemon = True
            p.start()
            self._file_queue.put_msg('reset')
            self._processes.append(p)
        self._downloading = True
        self._finished = 0

    def reset(self):
        for _ in range(self.num_downloader):
            self._file_queue.put_msg('reset')
        self._finished = 0

    def __del__(self):
        self.stop()

    def stop(self):
        for p in self._processes:
            self._file_queue.put_msg('Done')
        for p in self._processes:
            p.join()
        self._file_queue.close()
        self._downloading = False
