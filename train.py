import yaml
import argparse

# local import
from lib.trainer import Trainer


"""
Example call: 
python train.py -c my_model
"""
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default="model",
                    help='Checkpoint to load (if possible) and save a trained model')
args = parser.parse_args()

if __name__ == '__main__':
    CONFIG_FILE = "resources/train_config.yaml"

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    trainer = Trainer(
        config=config,
        model_name=args.model
    )
    trainer.loop()


