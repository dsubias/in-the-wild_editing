import argparse
import os
from agents import STGANAgent
from utils.config import *
import torch

def main():

    # Set resource usage
    torch.set_num_threads(8)
    os.nice(10)
    torch.cuda.empty_cache()

    # Parameter definitions
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-a', '--attribute', type=str, default='glossy', help='')   
    arg_parser.add_argument('-max', '--max_value', type=float, default = 1., help='') 
    arg_parser.add_argument('-min', '--min_value', type=float, default = 0.125, help='')  
    arg_parser.add_argument('-n', '--n_samples', type=int, default = 5, help='')  
    arg_parser.add_argument('-b', '--background', type=str, default='True', choices=['True', 'False'], help='') 
    arg_parser.add_argument('-v', '--video', type=str,  default='False', choices=['True', 'False'], help='') 
    
    # interval checks
    args = arg_parser.parse_args()
    assert args.min_value <= args.max_value 
    assert args.n_samples >= 1

    # str paramters to bool
    args.background = args.background == 'True'
    args.video = args.video == 'True'

    # mode selection
    if args.video:
        mode = 'edit_video'
    else: 

        mode = 'edit_images'

    # load backbone paramters
    config = process_config('configs/inference.yaml', mode)

    # update editing pameters
    if args.video:

        
        config.att_value_frame = args.max_value
        config.test_folder = './frames'
        config.num_samples = 1

    else: 

        config.test_folder = './test_images'
        config.num_samples = args.n_samples

    config.add_bg = args.background
    config.att_max = args.max_value
    config.att_min = args.min_value
    
    config.checkpoint = args.attribute
    config.cuda = False
    
    # run editing
    agent = STGANAgent(config)
    agent.run()


if __name__ == '__main__':
    main()
