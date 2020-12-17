from simpletransformers.classification import MultiLabelClassificationModel
import pandas as pd
import logging
import json

from preprocess_util import *
from sklearn.preprocessing import MultiLabelBinarizer
import random

random.seed(1234)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Train and Evaluation data needs to be in a Pandas Dataframe containing at least two columns, a 'text' and a 'labels' column. The `labels` column should contain multi-hot encoded lists.
with open("data/agent_intent_label.json") as in_file:
    label_list = json.load(in_file)
    label_list = [x for x in label_list.keys() if len(x) > 0]

print("Labels: ", len(label_list), "\n", label_list[:10], "...")

label_bin = MultiLabelBinarizer(classes=label_list)
label_bin.fit([])

def read_data(input_file_name):
    data_raw = read_data_for_act(input_file_name)
    random.shuffle(data_raw)

    text_a = [preprocess_history(x["history"]) for x in data_raw]
    #text_b = [preprocess_sys_usr(x["sys"], x["usr"]) for x in data_raw]
    label = label_bin.transform([x["act"] for x in data_raw])

    df = pd.DataFrame(zip(text_a, label), columns=["text", "labels"])
    print("Loaded %d data examples from %s" % (len(df), input_file_name))
    print(df.head())
    return df


train_df = read_data("data/train_with_act.json")

dev_df = read_data("data/dev_with_act.json")

# Create a MultiLabelClassificationModel
model_args = {
    'output_dir': 'outputs_act_new_label',
    'fp16': False, 
    'reprocess_input_data': False, 
    'overwrite_output_dir': True, 
    'num_train_epochs': 50,
    'max_seq_length': 512,
    'do_lower_case': True,
    'learning_rate': 1e-5,
    'use_early_stopping': True,
    'early_stopping_consider_epochs': True,
    'early_stopping_metric': "eval_loss",
    'early_stopping_metric_minimize': True,
    'early_stopping_patience': 5,
    'evaluate_during_training': True,
    'evaluate_during_training_verbose': True,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'process_count': 8
}

model = MultiLabelClassificationModel(
    'bert', 
    'model/ehealth-bert-joint-new/', 
    num_labels=len(label_list), 
    use_cuda=True, 
    args=model_args
)

# Train the model
model.train_model(train_df, eval_df=dev_df)


