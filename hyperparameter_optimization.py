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


def main(s_train, p_train, v_train):

    training_device = "cuda"

    ################################################################################
    # model parameters
    model = AZNeuralNetwork(
        game_size=3,
        model_parameters={"hidden_layers": [1024, 1024], "dropout": 0.0},
        inference_device=None
    )

    # training parameters
    epochs = 100000000
    patience = 100000000
    batch_size = 2048

    # optimization parameters
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.00001,
    )
    ################################################################################

    print(f"The dataset consist of {len(train_examples)} training examples (including augmentations).")

    s_batched, p_batched, v_batched = [], [], []
    for i in tqdm(range(0, len(train_examples), batch_size), file=stdout):
        s_batched.append(torch.tensor(np.vstack(s_train[i:i + batch_size]), dtype=torch.float32, device=training_device))
        p_batched.append(torch.tensor(np.vstack(p_train[i:i + batch_size]), dtype=torch.float32, device=training_device))
        v_batched.append(torch.tensor(v_train[i:i + batch_size], dtype=torch.float32, device=training_device))  # scalar v
    n_batches = len(s_batched)
    print(f"# batches: {n_batches}")

    current_patience = 0
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
        for i in tqdm(range(n_batches)):
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
                current_patience = 0
                print("New best model achieved!")
            else:
                current_patience += 1
                print("")


"""
- AlphaGo Zero: - 700,000 mini-batches of 2048 training examples (= 1,433,000,000 training examples, around 1.8 epochs then)
                - sampled from 500,000 games (from 20 iterations a 25,000) (a 200 moves = 100,000,000 training examples -> augmented: 800,000,000 training examples)
-           we: - 8,000 mini-batches of 2048 training examples (= 16,384,000 training examples, around 2 epochs then)
                - 16,000 mini-batches of 1024 training examples (= 16,384,000 training examples, around 2 epochs then)
                - 32,000 mini-batches of 512 training examples (= 16,384,000 training examples, around 2 epochs then)
                - sampled from 50,000 games (from 20 iterations a 2,500) (a 20 moves = 1,000,000 training examples -> augmented: 8,000,000 training examples)
- INFO: Only train for around 2 epochs per training loop iteration makes sense: each recorded data will be used in training of 20 iterations, while
  training data always improves in parallel (-> does not make sense to train longer with old data anyway)
"""
# TODO remove patience as it does not make sense anymore
if __name__ == '__main__':

    with open('data/optimization_data_augmented.json', 'r') as f:
        train_examples = json.load(f)
        train_examples = train_examples[:8000000]

    print(f"# train examples: {len(train_examples)}")
    train_examples = [(t["s"], t["p"], t["v"]) for t in train_examples]
    shuffle(train_examples)
    s_train, p_train, v_train = [list(t) for t in zip(*train_examples)]

    main(s_train, p_train, v_train)

