from sagemaker_algorithm_toolkit import metadata


def initialize(image_uri, hyperparameters, channels, metrics):
    training_spec = metadata.training_spec(
        hyperparameters, channels, metrics,
        image_uri,
        metadata.get_cpu_instance_types(metadata.Product.TRAINING),
        True)
    inference_spec = metadata.inference_spec(
        image_uri,
        metadata.get_cpu_instance_types(metadata.Product.HOSTING),
        metadata.get_cpu_instance_types(metadata.Product.BATCH_TRANSFORM),
        ["text/csv", "text/libsvm"],
        ["text/csv", "text/libsvm"])
    return metadata.generate_metadata(training_spec, inference_spec)
