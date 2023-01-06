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


def main(s, p, v):

    training_device = "cuda"

    # training parameters
    BATCH_SIZE = 1024
    mse_factor = 0.01

    """
    Data Prep using batch size.
    """
    print(f"The dataset consist of {len(train_examples)} training examples (including augmentations).")

    s_batched, p_batched, v_batched = [], [], []
    for i in tqdm(range(0, len(train_examples), BATCH_SIZE), file=stdout):
        s_batched.append(torch.tensor(np.vstack(s[i:i + BATCH_SIZE]), dtype=torch.float32, device=training_device))
        p_batched.append(torch.tensor(np.vstack(p[i:i + BATCH_SIZE]), dtype=torch.float32, device=training_device))
        v_batched.append(torch.tensor(v[i:i + BATCH_SIZE], dtype=torch.float32, device=training_device))  # scalar v

    n_batches = len(s_batched)
    n_batches_train = round(n_batches*0.9)
    n_batches_val = n_batches-n_batches_train
    print(f"# batches: {n_batches} / # train batches: {n_batches_train} / # val batches: {n_batches_val}")

    s_batched_train = s_batched[:n_batches_train]
    p_batched_train = p_batched[:n_batches_train]
    v_batched_train = v_batched[:n_batches_train]
    s_batched_val = s_batched[n_batches_train:]
    p_batched_val = p_batched[n_batches_train:]
    v_batched_val = v_batched[n_batches_train:]


    LEARNING_RATE = [0.01, 0.001, 0.0001]
    WEIGHT_DECAY = [0.001, 0.0001, 0.00001]
    DROPOUT = [0.0, 0.1, 0.2]
    HIDDEN_LAYERS = [[128, 128], [256, 256]]
    # HIDDEN_LAYERS = [[128, 128, 128], [256, 256, 256]]
    for learning_rate in LEARNING_RATE:
        for weight_decay in WEIGHT_DECAY:
            for dropout in DROPOUT:
                for hidden_layers in HIDDEN_LAYERS:
                    print(f"\n\n lr: {learning_rate} | weight_decay: {weight_decay} | dropout: {dropout} | layers: {hidden_layers}  (batch size fix: {BATCH_SIZE})")
                    ################################################################################
                    # model parameters

                    model = AZNeuralNetwork(
                        game_size=3,
                        model_parameters={"hidden_layers": hidden_layers, "dropout": dropout},
                        inference_device=None
                    )


                    # optimization parameters
                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=learning_rate,
                        momentum=0.9,
                        weight_decay=weight_decay,
                    )
                    ################################################################################


                    CrossEntropyLoss = torch.nn.CrossEntropyLoss()
                    MSELoss = torch.nn.MSELoss()

                    model.to(training_device)

                    for epoch in range(1, 12 + 1):

                        # train model
                        model.train()
                        for i in tqdm(range(n_batches_train), file=stdout):
                            optimizer.zero_grad()

                            p, v = model.forward(s_batched_train[i])

                            loss = CrossEntropyLoss(p, p_batched_train[i]) + mse_factor*MSELoss(v, v_batched_train[i])
                            loss.backward()
                            optimizer.step()

                        # evaluate model on train set
                        model.eval()
                        with torch.no_grad():
                            """
                            Train loss
                            """
                            # calculate loss per training example
                            loss = 0
                            for i in range(n_batches_train):
                                optimizer.zero_grad()
                                p, v = model.forward(s_batched_train[i])
                                loss += CrossEntropyLoss(p, p_batched_train[i]) + mse_factor*MSELoss(v, v_batched_train[i])
                            loss = loss / n_batches_train

                            print("[Epoch {0:d}] Train Loss: {1:.5f}. ".format(epoch, loss))


                            """
                            Val loss
                            """
                            # calculate loss per training example
                            loss = 0
                            for i in range(n_batches_val):
                                optimizer.zero_grad()
                                p, v = model.forward(s_batched_val[i])
                                loss += CrossEntropyLoss(p, p_batched_val[i]) + mse_factor*MSELoss(v, v_batched_val[i])
                            loss = loss / n_batches_val

                            print("[Epoch {0:d}] Val Loss: {1:.5f}. ".format(epoch, loss))



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

    n_total = 10000000

    with open('data/optimization_data_augmented.json', 'r') as f:
        train_examples = json.load(f)[:n_total]

    print(f"# train examples: {len(train_examples)}")
    train_examples = [(t["s"], t["p"], t["v"]) for t in train_examples]
    shuffle(train_examples)
    s, p, v = [list(t) for t in zip(*train_examples)]

    main(s, p, v)

