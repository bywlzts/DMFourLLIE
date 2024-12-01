import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .enhancement_model import enhancement_model as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
