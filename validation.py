from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.preprocessing.sequence import pad_sequences
import itertools
import matplotlib.pyplot as plt
import utils
import keras
import train
from sklearn.model_selection import train_test_split

FLAGS = train.FLAGS

dataframe = utils.load_dataset_with_random_corruption(FLAGS.targets)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# Data preprocessing
dataframe[0] = utils.encode_instruction_to_id(dataframe[0])
dataframe[1] = utils.encode_instruction_to_id(dataframe[1])

X = dataframe[[0, 1]]
Y = dataframe[2].astype(int)

_, X_validation, _, Y_validation = train_test_split(X, Y, test_size=0.33)

X_validation = {
    "left": X_validation[0],
    "right": X_validation[1],
}
# Zero padding
for dataset, side in itertools.product([X_validation], ["left", "right"]):
    dataset[side] = pad_sequences(dataset[side], maxlen=FLAGS.max_seq_len)

Y_validation = Y_validation.values

# Load a trained model
exponent_neg_manhattan_distance = utils.exponent_neg_manhattan_distance
innereye_bb = keras.models.load_model(FLAGS.saved_weights)

# Make predictions
pred = innereye_bb.predict([X_validation["left"], X_validation["right"]])

fpr, tpr, _ = roc_curve(Y_validation, pred, pos_label=1)
roc_auc = auc(fpr, tpr) * 100

try:
    print(f"AUC: {roc_auc_score(Y_validation, pred)}")
except:
    pass

plt.figure()
plt.plot(
    fpr,
    tpr,
    color="red",
    linewidth=1.2,
    label="Siamese Model (AUC = %0.2f%%)" % roc_auc,
)

plt.plot([0, 1], [0, 1], color="silver", linestyle=":", linewidth=1.2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()
