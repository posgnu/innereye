from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
import itertools
import utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
from time import time
import logging
import utils
import os
from sklearn.metrics import roc_auc_score
import json

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("proportion", 10, "Proportion of training data")
flags.DEFINE_list(
    "targets",
    ["curl", "openssl", "httpd", "sqlite3", "libcrypto"],
    "Target binary for training",
)
flags.DEFINE_integer("inst_embedding_dim", 100, "Dimension of instruction embeddings")
flags.DEFINE_integer(
    "n_units_2dn_layer", 50, "Number of hidden units in the second layer"
)
flags.DEFINE_integer(
    "n_units_1st_layer", 64, "Number of hidden units in the first layer"
)
flags.DEFINE_integer("max_seq_len", 101, "Maximum length of input sequences")
flags.DEFINE_string(
    "saved_weights",
    f'weights/siamese-LSTM_{"_".join(FLAGS.targets)}_{FLAGS.proportion}.h5',
    "Path of stored model weights",
)
flags.DEFINE_string(
    "saved_i2v_weights",
    f'embeddings/{FLAGS.inst_embedding_dim}D_minwordcount0_downsample1e-5_100epochs_{"_".join(FLAGS.targets)}_{FLAGS.proportion}.w2v',
    "Path of stored i2v weights",
)
flags.DEFINE_integer("epochs", 20, "Number of epochs")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    dataframe = utils.load_dataset_with_random_corruption(
        FLAGS.targets, num_of_neg_samples=2
    )
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    dataframe_for_i2v = utils.load_dataset(FLAGS.targets)

    # Train instruction2vec
    if not os.path.isfile(FLAGS.saved_i2v_weights):
        basic_block_stream = list(
            utils.block_level_instruction_stream(dataframe_for_i2v)
        )
        i2v_model = Word2Vec(
            sentences=basic_block_stream,
            vector_size=FLAGS.inst_embedding_dim,
            sample=1e-5,
            min_count=0,
        )
        i2v_model.train(
            basic_block_stream,
            total_examples=len(basic_block_stream),
            epochs=100,
        )
        i2v_model.save(FLAGS.saved_i2v_weights)

        print(i2v_model)
        print(i2v_model.wv.index_to_key[:10])
    else:
        # Load a trained instruction2vec model
        i2v_model = Word2Vec.load(FLAGS.saved_i2v_weights)

    embeddings, vocabulary = utils.build_embeddings_from_i2v(
        dataframe, i2v_model, FLAGS.inst_embedding_dim
    )

    # Store vocab
    with open(
        f'embeddings/vocab_{"_".join(FLAGS.targets)}_{FLAGS.proportion}.json', "w"
    ) as fp:
        json.dump(vocabulary, fp)

    # Train the siamese LSTMs
    X = dataframe[[0, 1]]
    Y = dataframe[2].astype(int)

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, Y, test_size=0.33
    )

    X_train = {"left": X_train[0], "right": X_train[1]}
    X_validation = {
        "left": X_validation[0],
        "right": X_validation[1],
    }

    # Zero padding
    for dataset, side in itertools.product([X_train], ["left", "right"]):
        dataset[side] = pad_sequences(dataset[side], maxlen=FLAGS.max_seq_len)
    for dataset, side in itertools.product([X_validation], ["left", "right"]):
        dataset[side] = pad_sequences(dataset[side], maxlen=FLAGS.max_seq_len)

    logging.info("Data preprocessing finished")

    Y_train = Y_train.values
    Y_validation = Y_validation.values

    # Build model
    n_epoch = FLAGS.epochs
    batch_size = 64

    # The visible layer
    left_input = Input(shape=(FLAGS.max_seq_len,), dtype="int32", name="input_1")
    right_input = Input(shape=(FLAGS.max_seq_len,), dtype="int32", name="input_2")

    embedding_layer = Embedding(
        len(embeddings),
        FLAGS.inst_embedding_dim,
        weights=[embeddings],
        input_length=FLAGS.max_seq_len,
        trainable=False,
    )

    # Embedded version of the inputs
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)

    # The 1st hidden layer
    shared_lstm_01 = LSTM(FLAGS.n_units_1st_layer, return_sequences=True)
    # The 2nd hidden layer
    shared_lstm_02 = LSTM(FLAGS.n_units_2dn_layer, activation="relu")

    left_output = shared_lstm_02(shared_lstm_01(encoded_left))
    right_output = shared_lstm_02(shared_lstm_01(encoded_right))

    malstm_distance = Lambda(
        lambda x: utils.exponent_neg_manhattan_distance(x[0], x[1]),
        output_shape=lambda x: (x[0][0], 1),
    )([left_output, right_output])

    # Pack it all up into a model
    innereye_bb = Model([left_input, right_input], [malstm_distance])

    innereye_bb.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.SGD(),
        metrics=["accuracy"],
    )
    innereye_bb.summary()

    # Start training
    training_start_time = time()

    innereye_bb_history = innereye_bb.fit(
        [X_train["left"], X_train["right"]],
        Y_train,
        batch_size=batch_size,
        epochs=n_epoch,
        validation_data=([X_validation["left"], X_validation["right"]], Y_validation),
    )

    training_end_time = time()
    print(
        "Training time finished.\n%d epochs in %12.2f"
        % (n_epoch, training_end_time - training_start_time)
    )

    innereye_bb.save(FLAGS.saved_weights)

    print("Training done.")

    # Make predictions
    pred = innereye_bb.predict([X_validation["left"], X_validation["right"]])

    print(f"AUC: {roc_auc_score(Y_validation, pred)}")
