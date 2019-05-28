from sagemaker_algorithm_toolkit import channel_validation as cv


def initialize():
    train_channel = cv.Channel(name="train", required=True)
    train_channel.add("csv", cv.Channel.FILE_MODE, cv.Channel.SHARDED)
    train_channel.add("csv", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
    train_channel.add("libsvm", cv.Channel.FILE_MODE, cv.Channel.SHARDED)
    train_channel.add("libsvm", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
    train_channel.add("text/csv", cv.Channel.FILE_MODE, cv.Channel.SHARDED)
    train_channel.add("text/csv", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
    train_channel.add("text/libsvm", cv.Channel.FILE_MODE, cv.Channel.SHARDED)
    train_channel.add("text/libsvm", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)

    validation_channel = cv.Channel(name="validation", required=False)
    validation_channel.add("csv", cv.Channel.FILE_MODE, cv.Channel.SHARDED)
    validation_channel.add("csv", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
    validation_channel.add("libsvm", cv.Channel.FILE_MODE, cv.Channel.SHARDED)
    validation_channel.add("libsvm", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
    validation_channel.add("text/csv", cv.Channel.FILE_MODE, cv.Channel.SHARDED)
    validation_channel.add("text/csv", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)
    validation_channel.add("text/libsvm", cv.Channel.FILE_MODE, cv.Channel.SHARDED)
    validation_channel.add("text/libsvm", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)

    # new for script mode/algorithm mode toggle
    code_channel = cv.Channel(name="code", required=False)
    code_channel.add("text/python", cv.Channel.FILE_MODE, cv.Channel.REPLICATED)

    return cv.Channels(train_channel, validation_channel, code_channel)
