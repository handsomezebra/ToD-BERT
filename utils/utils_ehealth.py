import json
import ast
import collections
import os

from .utils_function import get_input_example


def read_langs_turn(args, file_name, max_line=None, ds_name=""):
    print(("Reading from {} for read_langs_turn".format(file_name)))
    
    data = []
    
    with open(file_name) as f:
        dials = json.load(f)
        
    cnt_lin = 1
    for dial_dict in dials:
        dialog_history = []
        
        turn_usr = ""
        turn_sys = ""
        for ti, turn in enumerate(dial_dict["conversation"]):
            assert turn["role"] in ("User", "Agent")

            if turn["role"] == "User":
                turn_usr = " ".join(turn["text"]).lower().strip()
                
                data_detail = get_input_example("turn")
                data_detail["ID"] = "{}-{}".format(ds_name, cnt_lin)
                data_detail["turn_id"] = ti % 2
                data_detail["turn_usr"] = turn_usr
                data_detail["turn_sys"] = turn_sys
                data_detail["dialog_history"] = list(dialog_history)
                
                if not args["only_last_turn"]:
                    if len(turn["text"]) == 1 and 20 < len(turn["text"][0]) < 200:
                        data.append(data_detail)
                
                dialog_history.append(turn_sys)
                dialog_history.append(turn_usr)
                
            else:
                turn_sys = " ".join(turn["text"]).lower().strip()
        
        if args["only_last_turn"]:
            data.append(data_detail)
        
        cnt_lin += 1
        if(max_line and cnt_lin >= max_line):
            break

    return data


def read_langs_dial(file_name, ontology, dialog_act, max_line = None, domain_act_flag=False):
    print(("Reading from {} for read_langs_dial".format(file_name)))
    
    raise NotImplementedError



def prepare_data_ehealth(args):
    ds_name = "ehealth"
    
    example_type = args["example_type"]
    max_line = args["max_line"]
    
    file_trn = os.path.join(args["data_path"], "ehealth_train.json")
    file_dev = os.path.join(args["data_path"], "ehealth_dev.json")
    file_tst = os.path.join(args["data_path"], "ehealth_test.json")

    _example_type = "dial" if "dial" in example_type else example_type
    pair_trn = globals()["read_langs_{}".format(_example_type)](args, file_trn, max_line, ds_name)
    pair_dev = globals()["read_langs_{}".format(_example_type)](args, file_dev, max_line, ds_name)
    pair_tst = globals()["read_langs_{}".format(_example_type)](args, file_tst, max_line, ds_name)

    print("Read {} pairs train from {}".format(len(pair_trn), ds_name))
    print("Read {} pairs valid from {}".format(len(pair_dev), ds_name))
    print("Read {} pairs test  from {}".format(len(pair_tst), ds_name))  
    
    meta_data = {"num_labels":0}

    return pair_trn, pair_dev, pair_tst, meta_data

