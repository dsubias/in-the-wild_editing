import argparse
import os
from agents import STGANAgent
from utils.config import *
import torch

# Set resource usage
torch.set_num_threads(8)
os.nice(10)
torch.cuda.empty_cache()

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--config',
        default='configs/train_model.yaml',
        help='The path of configuration file in yaml format')
    args = arg_parser.parse_args()
    config = process_config(args.config)
    agent = STGANAgent(config)
    agent.run()


if __name__ == '__main__':
    main()
