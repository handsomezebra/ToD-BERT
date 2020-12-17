import logging

import pandas as pd
from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)

from preprocess_util import *


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = Seq2SeqArgs()
model_args.max_length = 40
model_args.num_beams = 5
model_args.num_train_epochs = 10
#model_args.no_save = True
#model_args.evaluate_generated_text = True
#model_args.evaluate_during_training = True
#model_args.evaluate_during_training_verbose = True

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
    use_cuda=True,
)

data = read_data_for_sum("data/test_with_act.json")

# Use the model for prediction

for d in data[:100]:
    print("=========================================================")
    text = preprocess_history(d, max_char_length=5000)
    print(text)
    text = text.replace("[CLS]", "").replace("[USR]", "").replace("[SYS]", "").replace("[SEP]", "")
    x = model.predict([text])[0]
    print("SUMMARY:", x)

