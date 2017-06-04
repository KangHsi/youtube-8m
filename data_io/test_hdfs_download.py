
import os
import sys
import tensorflow as tf
from hdfs_downloader import HDFSDownloader
import logging
import argparse
import threading

def test(data_pattern, save_dir, num_downloader=4, batch_size=4):
    downloader = HDFSDownloader(data_pattern, save_dir,
                    min_cache=batch_size, num_downloader=num_downloader)
    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)
    logging.info('Start downloading...')
    downloader.start_downloading()
    ret_code = 0
    dequeue_op = downloader.get_out_queue().dequeue_many(4)
    logging.info('Start testing...')

    while ret_code == 0:
        ret_code = downloader.enqueue(sess)
        outputs = sess.run([dequeue_op])
        deq_files = outputs[0]
        logging.info('Dequeue {} files: {}'.format(len(deq_files), deq_files))
        for f in deq_files:
            if os.path.isfile(f):
                os.remove(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-pattern', type=str,
                        default="",
                        help='Data pattern to download')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size to dequeue')
    parser.add_argument('--num-downloader', type=int, default=1,
                        help='Number of downloader')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Save dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format = '%(asctime)-15s %(message)s')
    if not args.data_pattern.startswith('hdfs://'):
        raise ValueError('Data pattern is not a HDFS file pattern')
    if not args.save_dir:
        args.save_dir = 'data_tmp'
        os.mkdir(args.save_dir)
    test(**vars(args))
