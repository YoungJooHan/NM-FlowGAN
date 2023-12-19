import argparse
from conf.default import get_cfg_defaults
from core.trainer import get_trainer_class
import os 

def main(config, device):
    # intialize trainer
    trainer = get_trainer_class(config.BASE.trainer)(config)
    trainer.set_device(device)
    trainer.train()


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default="./conf/default.yaml")
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument("opts", default=None,
                        help="Modify config options using the command-line",
                        nargs=argparse.REMAINDER,)
    
    args = parser.parse_args()
    config = get_cfg_defaults()

    if args.config != "":
        config.merge_from_file(args.config)
    if args.opts != "":
        config.merge_from_list(args.opts)
        
    main(config, args.device)


