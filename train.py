from lib.trainer import Trainer


def main():
    trainer = Trainer(size=3)
    trainer.train(
        num_epochs=10,
        num_plays=100,
        win_fraction=0.6
    )





if __name__ == '__main__':
    main()