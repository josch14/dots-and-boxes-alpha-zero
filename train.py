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
parser.add_argument('-id', '--inference_device', type=str, default="cpu", choices=["cpu", "cuda"],
                    help='Device with which model interference is performed during MCTS.')
parser.add_argument('-td', '--training_device', type=str, default="cuda", choices=["cpu", "cuda"],
                    help='Device with which model training is performed.')
args = parser.parse_args()

if __name__ == '__main__':
    CONFIG_FILE = "resources/train_config.yaml"

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    trainer = Trainer(
        config=config,
        model_name=args.model,
        inference_device=args.inference_device,
        training_device=args.training_device
    )
    trainer.loop()


