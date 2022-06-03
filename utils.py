import numpy as np
import keras.backend as K
import tensorflow as tf
import pandas as pd
import json
import logging

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS


def fix_disasm_dict(disasm_dict: dict) -> dict:
    key_list = [int(key) for key in disasm_dict.keys()]
    max_key = max(key_list)
    for i in range(max_key + 1):
        if i not in key_list:
            disasm_dict[str(i)] = []

    return disasm_dict


def encode_instruction_to_id(bb_list: list) -> list:
    with open(f'embeddings/vocab_{"_".join(FLAGS.targets)}.json', "r") as fp:
        vocab = json.load(fp)

        # bb as id list
        bb_id_list = []
        for bb in bb_list:
            bb_id_list.append([vocab[inst] if inst in vocab else 0 for inst in bb])

        return bb_id_list


def exponent_neg_manhattan_distance(left, right):
    """Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


# Generate instruction stream from datasets
def block_level_instruction_stream(dataset: list):
    """
    Extract instructions for making instruction2vec model.
    """
    for _, row in dataset.iterrows():
        yield row[0]
        yield row[1]


def random_corruption(alignments: np.ndarray, num_of_neg_samples: int) -> tuple:
    left_right = []
    right_left = []
    for i, _ in enumerate(alignments):
        indexing = np.ones((alignments.shape[0],), dtype=bool)
        indexing[i] = False
        left_right_candidate = np.random.choice(
            alignments[indexing, 1].flatten(), num_of_neg_samples
        )
        right_left_candidate = np.random.choice(
            alignments[indexing, 0].flatten(), num_of_neg_samples
        )

        left_right = np.hstack((left_right, left_right_candidate))
        right_left = np.hstack((right_left, right_left_candidate))

    return left_right, right_left


# Load data
def load_dataset(target: list) -> pd.Series:
    datasets = []
    for program in target:
        logging.info(f"Loading {program} dataset")
        if FLAGS.proportion == 10:
            csv_path = (
                "./data/done/" + program + "/innereye.csv"
            )
        elif FLAGS.proportion == 5:
            csv_path = (
                "./data/done/"
                + program
                + "/seed_alignments/5/training_alignments.csv"
            )
        else:
            raise NotImplemented

        disasm_path = (
            "./data/done/" + program + "/disasm_innereye.json"
        )

        if FLAGS.proportion == 10:
            dataframe = pd.read_csv(csv_path, header=None, dtype=object)
        elif FLAGS.proportion == 5:
            dataframe = pd.read_csv(csv_path, header=None, dtype=object)
            dataframe[2] = np.ones((dataframe.shape[0], 1))
        else:
            raise NotImplemented

        f = open(disasm_path, "r")
        disasm_dict = json.load(f)

        for index, row in dataframe.iterrows():
            for col in range(2):
                dataframe.at[index, col] = disasm_dict[str(row[col])]

        datasets.append(dataframe)

    return pd.concat(datasets, ignore_index=True)


# Load data with negative samples
def load_dataset_with_random_corruption(target: list, num_of_neg_samples=1) -> pd.Series:
    datasets = []
    for program in target:
        logging.info(f"Loading {program} dataset")

        if FLAGS.proportion == 10:
            csv_path = (
                "./data/done/" + program + "/innereye.csv"
            )
            alignment_path = (
                "./data/done/" + program + "/alignment.csv"
            )
        elif FLAGS.proportion == 5:
            csv_path = (
                "./data/done/"
                + program
                + "/seed_alignments/5/training_alignments.csv"
            )
            alignment_path = (
                "./data/done/"
                + program
                + "/seed_alignments/5/training_alignments.csv"
            )
        else:
            raise NotImplemented
        disasm_path = (
            "./data/done/" + program + "/disasm_innereye.json"
        )

        if FLAGS.proportion == 10:
            dataframe = pd.read_csv(csv_path, header=None, dtype=object)
        elif FLAGS.proportion == 5:
            dataframe = pd.read_csv(csv_path, header=None, dtype=object)
            dataframe[2] = np.ones((dataframe.shape[0], 1))
        else:
            raise NotImplemented

        alignments = pd.read_csv(alignment_path, header=None)
        f = open(disasm_path, "r")
        disasm_dict = json.load(f)

        # Random corruption
        numpy_dataframe = dataframe.to_numpy().astype(float)
        train_data_len = dataframe.shape[0]

        L = np.ones((train_data_len, num_of_neg_samples)) * (
            numpy_dataframe[:, 0].reshape((train_data_len, 1))
        )
        negative_samples_left_left = L.reshape((train_data_len * num_of_neg_samples, 1))

        L = np.ones((train_data_len, num_of_neg_samples)) * (
            numpy_dataframe[:, 1].reshape((train_data_len, 1))
        )
        negative_samples_right_right = L.reshape(
            (train_data_len * num_of_neg_samples, 1)
        )

        negative_samples_left_right, negative_samples_right_left = random_corruption(
            alignments.to_numpy(), num_of_neg_samples
        )

        negative_samples_left_right = negative_samples_left_right.reshape(
            (train_data_len * num_of_neg_samples, 1)
        )

        negative_samples_right_left = negative_samples_right_left.reshape(
            (train_data_len * num_of_neg_samples, 1)
        )

        negative_samples_label = np.zeros((train_data_len * num_of_neg_samples, 1))

        negative_samples_left = np.hstack(
            (
                negative_samples_left_left,
                negative_samples_left_right,
                negative_samples_label,
            )
        )
        negative_samples_right = np.hstack(
            (
                negative_samples_right_left,
                negative_samples_right_right,
                negative_samples_label,
            )
        )

        dataframe = pd.concat(
            [
                dataframe,
                pd.DataFrame(negative_samples_left),
                pd.DataFrame(negative_samples_right),
            ],
            ignore_index=True,
        )

        for index, row in dataframe.iterrows():
            for col in range(2):
                dataframe.at[index, col] = disasm_dict[str(int(row[col]))]

        datasets.append(dataframe)

    return pd.concat(datasets, ignore_index=True)


def build_embeddings_from_i2v(dataset, i2v, inst_embedding_dim):
    vocabulary = dict()
    # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
    inverse_vocabulary = ["<unk>"]

    # Iterate over the basic blocks of both training and test datasets
    for index, row in dataset.iterrows():
        # Iterate through the text of both basic blocks of the row
        for col in range(2):
            q2n = []  # q2n -> basic block numbers representation
            for instruction in row[col]:
                # Check for unwanted instructions
                if instruction not in i2v.wv:
                    print("Unknown instruction is found!!!")
                    continue
                if instruction not in vocabulary:
                    vocabulary[instruction] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(instruction)
                else:
                    q2n.append(vocabulary[instruction])
            # Replace basic block as insturction to basic block as number representation
            dataset.at[index, col] = q2n

    # This will be the embedding matrix
    embeddings = 1 * np.random.randn(len(vocabulary) + 1, inst_embedding_dim)
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabulary.items():
        if word in i2v.wv:
            embeddings[index] = i2v.wv[word]
        else:
            raise ValueError("Unknown instruction is found!!!")

    return embeddings, vocabulary
