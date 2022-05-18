import os
import yaml
import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
from easydict import EasyDict

from utils.misc import create_dirs

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def setup_logging(log_dir):
    log_file_format = '[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d'
    log_console_format = '[%(levelname)s]: %(message)s'

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler(
        '{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler(
        '{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as config_file:
        try:
            config_dict = yaml.load(config_file, Loader=Loader)
            file = config_dict
            config = EasyDict(config_dict)
            return config,file
        except ValueError:
            print('INVALID YAML file format.. Please provide a good yaml file')
            exit(-1)


def process_config(yaml_file):
    config, file = get_config_from_yaml(yaml_file)

    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' The experiment mode is {} '.format(config.mode))
    print(' *************************************** ')

    # create some important directories to be used for that experiments
    config.summary_dir = os.path.join(config.out_root,
        'experiments', config.exp_name, 'summaries/')
    config.checkpoint_dir = os.path.join(config.out_root,
        'experiments', config.exp_name, 'checkpoints/')
    config.sample_dir = os.path.join(config.out_root,
        'experiments', config.exp_name, 'samples/')
    config.log_dir = os.path.join(config.out_root,'experiments', config.exp_name, 'logs/')
    config.result_dir = os.path.join(config.out_root,
        'experiments', config.exp_name, 'results/')

    dir_list = [config.summary_dir, config.checkpoint_dir,
                config.sample_dir, config.log_dir, config.result_dir]
    if config.video:
        config.video_dir = os.path.join(config.out_root,'experiments', config.exp_name, 'video/')
        dir_list.append(config.video_dir)

    create_dirs(dir_list)
    if config.histogram:
        config.histogram_dir = os.path.join(config.out_root,
        'experiments', config.exp_name, 'results/histograms')
        if not os.path.exists(config.histogram_dir):
                os.makedirs(config.histogram_dir)
    if config.mode == 'test' and config.plot_metrics:
        config.metric_dir = os.path.join(
        'experiments', config.exp_name, 'results/metrics')
        if not os.path.exists(config.metric_dir):
                os.makedirs(config.metric_dir)
    if config.mode=='train':
        with open(os.path.join(config.out_root,'experiments', config.exp_name,'summary.yaml'), 'w') as f:
            yaml.dump(file, f)
    

    # setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info('Hi, This is root.')
    logging.getLogger().info(
        'After the configurations are successfully processed and dirs are created.')
    logging.getLogger().info('The pipeline of the project will begin now.')

    return config
