import argparse, os, logging
from configs.config import MainConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    
    parser.add_argument("-c", "--config", help="The path of config file.", type=str)

    return parser.parse_args()


def config_log(cfg):
    FORMAT = '%(asctime)s, %(message)s'
    logging.basicConfig(
        filename=os.path.join(cfg.result_dir, cfg.nick_name+'.txt'),
        level=logging.INFO,
        filemode='w',
        format=FORMAT
    )
    logging.getLogger().addHandler(logging.StreamHandler())





def main(args):
    
    cfg = MainConfig(args.config)
    


    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)