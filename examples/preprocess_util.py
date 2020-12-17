import csv
import json
import torch
import os
from collections import Counter

import random

random.seed(1234)

ignore_categories = set([
    "Agent/Agent Action/Misc/agent_say_ok",
])


def read_data_for_act(data_file_name, max_history=10):
    print(("Reading data from {}".format(data_file_name)))
    
    with open(data_file_name) as f:
        dials = json.load(f)
        
    data = []
    for dial_dict in dials:
        dialog_history = []
        
        turn_usr = ""
        turn_sys = ""
        cur_sys_act = []
        for ti, turn in enumerate(dial_dict["conversation"]):
            assert turn["role"] in ("User", "Agent")

            if turn["role"] == "Agent":
                turn_sys = " ".join(turn["text"]).lower().strip()
                cur_sys_act = turn["act"]
                cur_sys_act = [x for x in cur_sys_act if x not in ignore_categories]
                cur_sys_act = list(set(cur_sys_act))
            else:
                turn_usr = " ".join(turn["text"]).lower().strip()
                
                data_detail = {}
                data_detail["sys"] = turn_sys
                data_detail["usr"] = turn_usr
                data_detail["history"] = list(dialog_history)
                data_detail["act"] = cur_sys_act
                
                if len(cur_sys_act) > 1:
                    r = random.randint(1, 100)
                    if r <= 40:
                        data.append(data_detail)
                    cur_sys_act = []
                elif len(cur_sys_act) == 1:
                    r = random.randint(1, 100)
                    if r <= 20:
                        data.append(data_detail)
                    cur_sys_act = []
                else:
                    r = random.randint(1, 100)
                    if r <= 1:  # 1% of examples without class labels
                        data.append(data_detail)
                    cur_sys_act = []
                
                dialog_history.append(turn_sys)
                dialog_history.append(turn_usr)
                dialog_history = dialog_history[-max_history:]

    return data


def read_data_for_rs(data_file_name, max_history=10):
    print(("Reading data from {}".format(data_file_name)))
    
    with open(data_file_name) as f:
        dials = json.load(f)
        
    sys_response_sample = Counter()
    data = []
    for dial_dict in dials:
        dialog_history = []
        
        turn_usr = ""
        turn_sys = ""
        for ti, turn in enumerate(dial_dict["conversation"]):
            assert turn["role"] in ("User", "Agent")

            if turn["role"] == "Agent":
                turn_sys = " ".join(turn["text"]).lower().strip()
            else:
                turn_usr = " ".join(turn["text"]).lower().strip()
                
                data_detail = {}
                data_detail["sys"] = turn_sys
                data_detail["usr"] = turn_usr
                data_detail["history"] = list(dialog_history)
                
                if 20 < len(turn_usr) < 200 and 20 < len(turn_sys) < 200:
                    data.append(data_detail)

                if 20 < len(turn_sys) < 200:
                    sys_response_sample[turn_sys] += 1
                
                dialog_history.append(turn_sys)
                dialog_history.append(turn_usr)
                dialog_history = dialog_history[-max_history:]

    sys_response_sample = sys_response_sample.most_common()
    sys_response_sample = [x[0] for x in sys_response_sample if x[1] > 1]
    print("%d responses collected for negative examples" % len(sys_response_sample))
    return data, sys_response_sample

def read_data_for_rs_intent(data_file_name, max_history=10):
    print(("Reading data from {}".format(data_file_name)))
    
    with open(data_file_name) as f:
        dials = json.load(f)
        
    data = []
    for dial_dict in dials:
        dialog_history = []
        
        turn_usr = ""
        turn_sys = ""
        cur_sys_act = []
        for ti, turn in enumerate(dial_dict["conversation"]):
            assert turn["role"] in ("User", "Agent")

            if turn["role"] == "Agent":
                turn_sys = " ".join(turn["text"]).lower().strip()
                cur_sys_act = turn["act"]
            else:
                turn_usr = " ".join(turn["text"]).lower().strip()
                
                data_detail = {}
                data_detail["sys"] = turn_sys
                data_detail["usr"] = turn_usr
                data_detail["history"] = list(dialog_history)
                data_detail["act"] = cur_sys_act
                
                if len(cur_sys_act) == 1:
                    data.append(data_detail)
                    cur_sys_act = []
                
                dialog_history.append(turn_sys)
                dialog_history.append(turn_usr)
                dialog_history = dialog_history[-max_history:]

    return data

def read_data_for_sum(data_file_name):
    print(("Reading data from {}".format(data_file_name)))
    
    with open(data_file_name) as f:
        dials = json.load(f)
        
    data = []
    for dial_dict in dials:
        dialog_history = []
        
        turn_usr = ""
        turn_sys = ""
        for ti, turn in enumerate(dial_dict["conversation"]):
            assert turn["role"] in ("User", "Agent")

            if turn["role"] == "Agent":
                turn_sys = " ".join(turn["text"]).lower().strip()
            else:
                turn_usr = " ".join(turn["text"]).lower().strip()
                
                dialog_history.append(turn_sys)
                dialog_history.append(turn_usr)
                turn_usr = ""
                turn_sys = ""
        data.append(dialog_history)

    return data


def read_sentences(data_file_name):
    print(("Reading data from {}".format(data_file_name)))
    
    with open(data_file_name) as f:
        dials = json.load(f)
        
    data = []
    for dial_dict in dials:
        agent_history = []
        customer_history = []
        
        for ti, turn in enumerate(dial_dict["conversation"]):
            assert turn["role"] in ("User", "Agent")

            if turn["role"] == "Agent":
                for t in turn["text"]:
                    agent_history.append(t)
            else:
                for t in turn["text"]:
                    customer_history.append(t)

        data.append({"agent": agent_history, "customer": customer_history})

    return data

def get_labels(sentence_label_dict, sentence_list):

    labels = []
    for sentence in sentence_list:
        sentence = sentence.lower().strip()

        if sentence in sentence_label_dict:
            labels.append(sentence_label_dict[sentence])

    return labels


cls_token = "[CLS]"
sys_token = "[SYS]"
sep_token = "[SEP]"
usr_token = "[USR]"

def preprocess(dialog_history, sys, usr=None):
    """Converts history and utterance to sequences
    The format of the result sequence is like:
    "[CLS] [SYS] ... [USR] ... [SEP] [SYS] now i want to understand [USR] yeah, i have"
    or if usr is empty
    "[CLS] [SYS] ... [USR] ... [SEP] [SYS] now i want to understand"

    """

    if usr is not None:
        text = " %s %s %s %s %s" % (sep_token, sys_token, sys, usr_token, usr)
    else:
        text = " %s %s %s" % (sep_token, sys_token, sys)

    history_text = preprocess_history(dialog_history)

    text = "[CLS] " + history_text + text

    return text

def preprocess_sys_usr(sys, usr):
    text = "%s %s %s %s" % (sys_token, sys, usr_token, usr)

    return text

def preprocess_history(dialog_history, max_char_length=1800):
    """Converts history to sequences
    The format of the result sequence is like:
    "[SYS] ... [USR] ... [SYS] now i want to understand [USR] yeah, i have"

    """
    assert len(dialog_history) % 2 == 0  # contains pairs of sys and usr utterances

    text = ""

    for idx, utt in enumerate(reversed(dialog_history)):
        if idx % 2 == 0:
            role = usr_token
        else:
            role = sys_token

        new_text = "%s %s " % (role, utt) + text

        if len(new_text) > max_char_length:
            # about 500 tokens
            break

        text = new_text

    return text.strip()

def tokenize(tokenizer, text, max_length=512):
    if text.startswith(cls_token + " "):
        text = text[6:]

    tokens = tokenizer.tokenize(cls_token) + tokenizer.tokenize(text)[-max_length+1:]
    token_ids = torch.Tensor(tokenizer.convert_tokens_to_ids(tokens))

    return token_ids
