from sklearn.externals import joblib


def load_pipeline(path):
    """
    Loads sci-kit learn pipeline from persistent storage.

    :param path: Path to the sci-kit learn pipeline.
    :return: pipeline to save
    """
    pipeline = joblib.load('{}.pkl'.format(path))
    if hasattr(pipeline.steps[-1][-1], 'load'):
        pipeline.steps[-1][-1].load(path)
    return pipeline


def save_pipeline(path, pipeline):
    """
    Saves sci-kit learn pipeline to persistent storage

    :param path: Path to the sci-kit learn pipeline.
    :param pipeline: pipeline to save
    :return:
    """
    joblib.dump(pipeline, '{}.pkl'.format(path))
    if hasattr(pipeline.steps[-1][-1], 'save'):
        pipeline.steps[-1][-1].save(path)
