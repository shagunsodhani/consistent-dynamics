import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description="Argument parser to obtain the name of the config file")
    parser.add_argument('--config_id', default="debug", help='config id to use')
    args = parser.parse_args()
    return args.config_id