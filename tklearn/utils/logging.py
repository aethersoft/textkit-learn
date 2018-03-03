import logging

__all__ = ['build_logger']


def build_logger(name: str,
                 output_format: str = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                 log_level=logging.INFO) -> logging.Logger:
    """
    Builds logger and initialize it with the name provided
    :return: initialized logger
    """
    output = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(output_format)
    handler.setFormatter(formatter)
    output.addHandler(handler)
    output.setLevel(log_level)
    return output
