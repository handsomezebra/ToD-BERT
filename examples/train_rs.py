from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.ERROR)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

train_file_name = "data/train.tsv"

eval_file_name = "data/dev.tsv"


model_args = {
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
    'evaluate_during_training_steps': 20000,
    'evaluate_during_training_verbose': True,
    'train_batch_size': 8,
    'eval_batch_size': 8,
    'lazy_loading': True,
    'lazy_text_a_column': 0,
    'lazy_text_b_column': 1,
    'lazy_labels_column': 2
}

model = ClassificationModel(
    'bert', 
    'model/ehealth-bert-joint-new/', 
    use_cuda=True, 
    args=model_args
)

# Train the model
model.train_model(train_file_name, eval_df=eval_file_name)


