from keras.preprocessing.sequence import pad_sequences
import utils
import keras
import train
from keras.models import Model
import json
import numpy as np
import torch

FLAGS = train.FLAGS

# Load a trained model
exponent_neg_manhattan_distance = utils.exponent_neg_manhattan_distance
innereye_bb = keras.models.load_model(FLAGS.saved_weights)

lstm = Model(inputs=innereye_bb.input[0], outputs=innereye_bb.layers[4].output)

if __name__ == "__main__":
    for program in FLAGS.targets:
        disasm_path = (
            utils.get_git_root("extract.py")
            + "/data/done/"
            + program
            + "/disasm_innereye.json"
        )
        embedding_file_path = (
            utils.get_git_root("extract.py")
            + "/data/done/"
            + program
            + f'/innereye_embeddings_{"_".join(FLAGS.targets)}_{FLAGS.proportion}.pt'
        )
        f = open(disasm_path, "r")
        disasm_dict = json.load(f)
        disasm_dict = utils.fix_disasm_dict(disasm_dict)

        bb_list = []
        for id in range(len(disasm_dict)):
            bb_list.append(disasm_dict[str(id)])
        # Data preprocessing
        bb_list = utils.encode_instruction_to_id(bb_list)
        bb_list = pad_sequences(np.array(bb_list), maxlen=FLAGS.max_seq_len)

        lstm_output = lstm.predict(bb_list)

        torch.save(torch.tensor(lstm_output), embedding_file_path)
