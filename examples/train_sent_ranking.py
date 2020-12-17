from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import random
import json

random.seed(1234)


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
data = []

with open("data/save_list_sent.json") as in_file:
    save_list_data = json.load(in_file)

for d in save_list_data:
    for s in d["sentences"]:
        data.append((s.lower(), 1))

print("%d positive numbers" % len(data))

with open("data/sentence_list.json") as in_file:
    sentence_list = json.load(in_file)

sentence_list = [x[0] for x in sentence_list]
sentence_list = [x for x in sentence_list if len(x) > 20]
sentence_list = sentence_list[:50000]
random.shuffle(sentence_list)
sentence_list = sentence_list[:5000]
print("%d negative numbers" % len(sentence_list))

for s in sentence_list:
    data.append((s, 0))

random.shuffle(data)

data_len = len(data)
train_data_len = int(data_len * 0.85)

train_data = data[:train_data_len]
eval_data = data[train_data_len:]

#train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
#eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
train_df = pd.DataFrame(train_data)
eval_df = pd.DataFrame(eval_data)

print("Train:", train_df)
print("Eval:", eval_df)

model_args = {
    'output_dir': 'outputs_sent_ranking_with_tod',
    'fp16': False, 
    'reprocess_input_data': False, 
    'overwrite_output_dir': True, 
    'num_train_epochs': 50,
    'max_seq_length': 512,
    'learning_rate': 1e-5,
    'do_lower_case': True,
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

# Create a ClassificationModel
model = ClassificationModel('bert', 'model/ehealth-bert-joint-new/', args=model_args)

# Train the model
model.train_model(train_df, eval_df=eval_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

