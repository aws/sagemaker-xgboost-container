#!/usr/bin/env python

"""
Original source: https://github.com/ishaaq/sagemaker-pipe
Author: ishaaq

This project builds a very simple implementation of SageMaker Training's internal IO subsystem that
is able to pipe channel data files to an algorithm. It is meant to be used as a local-testing tool
in order to test a PIPE mode algorithm locally before attempting to run it for real with SageMaker Training.
"""

import argparse
import errno
import gzip
import logging
import os
import shutil
import stat
from urllib.parse import urlparse

import boto3

NUM_EPOCHS = 5


def run(args):
    src = args.src
    dest = args.dest
    channel = args.channel
    print("Pipe from src: {} to dest: {} for channel: {}".format(src, dest, channel))

    if src.startswith("s3://"):
        s3_uri = urlparse(src)
        bucket_str = s3_uri.netloc
        prefix = s3_uri.path.lstrip("/")
        logging.debug("bucket: {}, prefix: {}".format(bucket_str, prefix))
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(bucket_str)

        def src_retriever(sink):
            s3_retriever(bucket, prefix, sink)

    else:

        def src_retriever(sink):
            local_retriever(src, sink)

    if args.gunzip:

        def unzipper(sink):
            tmp_path = dest + "/." + channel + ".tmp"
            gunzip(src_retriever, tmp_path, sink)
            os.unlink(tmp_path)

        run_pipe(channel, unzipper, dest)
    else:
        run_pipe(channel, src_retriever, dest)


def s3_retriever(bucket, prefix, sink):
    for obj_summary in bucket.objects.filter(Prefix=prefix):
        logging.debug("streaming s3://{}/{}".format(bucket.name, obj_summary.key))
        bucket.download_fileobj(obj_summary.key, sink)


def local_retriever(src, sink):
    if os.path.isfile(src):
        logging.debug("streaming file: {}".format(src))
        with open(src, "rb") as src:
            shutil.copyfileobj(src, sink)
    else:
        for root, dirs, files in os.walk(src):
            logging.debug("file list: {}".format(files))
            for file in files:
                src_path = root + "/" + file
                logging.debug("streaming file: {}".format(src_path))
                if os.path.isfile(src_path):  # ignore special files
                    with open(src_path, "rb") as src:
                        shutil.copyfileobj(src, sink)


def gunzip(src_retriever, tmp_path, sink):
    with open(tmp_path, "wb") as tmp:
        src_retriever(tmp)
    with gzip.open(tmp_path, "rb") as inflated:
        shutil.copyfileobj(inflated, sink)


def run_pipe(channel, src_retriever, dest):
    """Emulate SageMaker pipe.

    :param channel: Pipe name
    :param src_retriever: Function to stream data to the pipe
    :param dest: Path to directory of the pipe
    """
    for epoch in range(NUM_EPOCHS):
        print("Running epoch: {}".format(epoch))
        # delete previous epoch's fifo if it exists:
        delete_fifo(dest, channel, epoch - 1)

        try:
            fifo_pth = create_fifo(dest, channel, epoch)
            with open(fifo_pth, mode="bw", buffering=0) as fifo:
                src_retriever(fifo)
        except IOError as e:
            if e.errno == errno.EPIPE:
                print("Client closed current epoch's pipe before reaching EOF. " "Continuing with next epoch...")
            else:
                raise
        finally:
            delete_fifo(dest, channel, epoch)
    print("Completed pipe for channel: {}".format(channel))


def fifo_path(dest, channel, epoch):
    """Construct full FIFO path.

    :param dest: Path to directory of FIFO
    :param channel: FIFO name
    :param epoch: Integer appended to FIFO name indicating the training epoch
    :return: full FIFO path
    """
    return dest + "/" + channel + "_" + str(epoch)


def delete_fifo(dest, channel, epoch):
    """Delete FIFO.

    :param dest: Path to directory of FIFO
    :param channel: FIFO name
    :param epoch: Integer appended to FIFO name indicating the training epoch
    """
    try:
        path = fifo_path(dest, channel, epoch)
        os.unlink(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            # if the fifo file doesn't exist we don't care, we were going to
            # delete it anyway, otherwise raise:
            raise


def create_fifo(dest, channel, epoch):
    """Create FIFO.

    :param dest: Path of directory to create FIFO in
    :param channel: FIFO name
    :param epoch: Integer appended to FIFO name indicating the training epoch
    :return: Boolean, True if path is FIFO, else False
    """
    path = fifo_path(dest, channel, epoch)
    logging.debug("Creating fifo: {}".format(path))
    mkdir(os.path.dirname(path))
    if not is_fifo(path):
        os.mkfifo(path)
    return path


def is_fifo(path):
    """Check if the specified path is a FIFO.

    :param path: Path of FIFO to check
    :return: Boolean, True if path is FIFO, else False
    """
    if not os.path.isfile(path):
        return False
    return stat.S_ISFIFO(os.stat(path).st_mode)


def mkdir(path):
    """Create the specified directory.

    :param path: Path of the directory to create
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
A local testing tool for algorithms that use SageMaker Training in
PIPE mode.
""",
        epilog="""
Examples:
> sagemaker-pipe.py training src-dir dest-dir
The above example will recursively walk through all the files under
src-dir and stream their contents into FIFO files named:
dest-dir/training_0
dest-dir/training_1
dest-dir/training_2
...
> sagemaker-pipe.py train s3://mybucket/prefix dest-dir
This example will recursively walk through all the objects under
s3://mybucket/prefix and similarly stream them into FIFO files:
dest-dir/train_0
dest-dir/train_1
dest-dir/train_2
...
Note that for the above to work the tool needs credentials. You can
set that up either via AWS credentials environment variables:
https://boto3.readthedocs.io/en/latest/guide/configuration.html#environment-variables
OR via a shared credentials file:
https://boto3.readthedocs.io/en/latest/guide/configuration.html#aws-config-file
""",
    )

    parser.add_argument("-d", "--debug", action="store_true", help="enable debug messaging")
    parser.add_argument("-x", "--gunzip", action="store_true", help="inflate gzipped data before streaming it")
    parser.add_argument("-r", "--recordio", action="store_true", help="wrap individual files in recordio records")
    parser.add_argument("channel", metavar="CHANNEL_NAME", help="the name of the channel")
    parser.add_argument("src", metavar="SRC", help="the source, can be an S3 uri or a local path")
    parser.add_argument(
        "dest",
        metavar="DEST",
        help="the destination dir where the data is to be \
                        streamed to",
    )
    args, unknown = parser.parse_known_args()

    if unknown:
        logging.warning("Ignoring unknown arguments: {}".format(unknown))
    logging.debug("Training with configuration: {}".format(args))

    if args.debug:
        logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.DEBUG)

    if args.recordio:
        logging.warning("recordio wrapping not implemented yet - ignoring!")

    run(args)


if __name__ == "__main__":
    main()
