import argparse
import imp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', 
        type=str, default='./args_cos.py',
        help="The path to the configuration file")
    args = parser.parse_args()

    config = imp.load_source('config', args.config_file)

    return config
