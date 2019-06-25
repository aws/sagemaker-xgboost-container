from sagemaker_algorithm_toolkit import exceptions


def get_content_type(request):
    content_type = request.content_type or "text/csv"
    content_type = content_type.lower()
    tokens = content_type.split(";")
    content_type = tokens[0].strip()
    if content_type not in ['text/csv', 'text/libsvm', 'text/x-libsvm']:
        raise exceptions.UserError("Content-type {} not supported. "
                                   "Supported content-type is text/csv, text/libsvm"
                                   .format(content_type))
    return content_type
