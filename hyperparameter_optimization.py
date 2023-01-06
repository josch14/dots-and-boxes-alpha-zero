import json
import time
from copy import deepcopy
from random import shuffle
from sys import stdout

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src import AZNeuralNetwork, DotsAndBoxesGame


def main(config, s_train, p_train, v_train):

    training_device = "cuda"

    model_parameters = config["model_parameters"]
    optimizer_parameters = config["optimizer_parameters"]
    training_parameters = config["training_parameters"]

    model = AZNeuralNetwork(
        game_size=3,
        model_parameters=model_parameters,
        inference_device=None
    )

    # parameters
    game_buffer_size = training_parameters["game_buffer_size"]
    epochs = training_parameters["epochs"]
    batch_size = training_parameters["batch_size"]
    patience = training_parameters["patience"]

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optimizer_parameters["learning_rate"],
        momentum=optimizer_parameters["momentum"],
        weight_decay=optimizer_parameters["weight_decay"],
    )

    print(f"The dataset consist of {len(train_examples)} training examples (including augmentations).")

    # TODO "Parameters were updated from 700,000 mini-batches of 2,048 positions. ...
    # TODO ... Each mini-batch of data is  sampled uniformly at random from all positions of the most recent 500,000 games (a 200 moves = 10M examples)
    #  (we: 100,000 games a 18 moves = 1.8M)
    # TODO of self-play"

    s_batched, p_batched, v_batched = [], [], []
    for i in tqdm(range(0, len(train_examples), batch_size), file=stdout):
        s_batched.append(torch.tensor(np.vstack(s_train[i:i + batch_size]), dtype=torch.float32, device=training_device))
        p_batched.append(torch.tensor(np.vstack(p_train[i:i + batch_size]), dtype=torch.float32, device=training_device))
        v_batched.append(torch.tensor(v_train[i:i + batch_size], dtype=torch.float32, device=training_device))  # scalar v
    n_batches = len(s_batched)

    current_patience = 0
    best_model = None
    best_loss = 1e10

    CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    MSELoss = torch.nn.MSELoss()

    model.to(training_device)

    for epoch in range(1, epochs + 1):

        if current_patience > patience:
            print(f"Early stopping after {epoch} epochs.")
            break

        start_time = time.time()

        # train model
        model.train()
        for i in range(n_batches):
            optimizer.zero_grad()

            p, v = model.forward(s_batched[i])

            loss = CrossEntropyLoss(p, p_batched[i]) + MSELoss(v, v_batched[i])
            loss.backward()
            optimizer.step()

        # evaluate model on train set
        model.eval()
        with torch.no_grad():
            # calculate loss per training example
            loss = 0
            for i in range(n_batches):
                optimizer.zero_grad()
                p, v = model.forward(s_batched[i])
                loss += CrossEntropyLoss(p, p_batched[i]) + MSELoss(v, v_batched[i])
            loss = loss / n_batches

            print("[Epoch {0:d}] Loss: {1:.5f} (after {2:.2f}s). ".format(epoch, loss, time.time() - start_time), end="")

            if loss < best_loss:
                best_loss = loss
                best_model = deepcopy(model)
                current_patience = 0
                print("New best model achieved!")
            else:
                current_patience += 1
                print("")

    model = best_model




if __name__ == '__main__':
    CONFIG_FILE = "resources/train_config.yaml"

    with open('data/optimization_data.json', 'r') as f:
        train_examples = json.load(f)
        print(type(train_examples))

    print(f"# train examples: {len(train_examples)}")
    print("extracting dict ..")
    train_examples = [(t["s"], t["p"], t["v"]) for t in train_examples]
    shuffle(train_examples)
    s_train, p_train, v_train = [list(t) for t in zip(*train_examples)]

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    main(config, s_train, p_train, v_train)

