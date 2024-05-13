from chatbot import models, wrappers, retrievers, cli

import yaml
import os
import threading
import argparse
import sqlite3
import pandas as pd
import time
from importlib import import_module

def get_cls(dotpath: str):
    """load object from module."""
    module_, func = dotpath.rsplit(".", maxsplit=1)
    m = import_module(module_)
    return getattr(m, func)

def get_args():
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument("yaml_path")
    return parser.parse_args()

def main():
    argv = get_args()

    with open(argv.yaml_path, 'r') as file:
        configs = yaml.safe_load(file)

    cnx = sqlite3.connect(configs["db"]["path"])
    cmd = configs["db"]["command"]
    df = pd.read_sql_query(cmd, cnx)

    ret_cls = get_cls(configs["retriever"]["cls"])
    bot_cls = get_cls(configs["model"]["cls"])

    indexing_fn = eval(configs["retriever"]["indexing_fn"])
    ret = ret_cls(df, indexing_fn(df))  #HACK exec is a bad practice... find a solution

    bot_kwargs = configs["model"].get("kwargs", {})
    bot = bot_cls(df, ret, **bot_kwargs)

    for wrapper_cls_name, args, kwargs in configs["wrappers"]:
        wrapper = get_cls(wrapper_cls_name)
        bot = wrapper(bot, *args, **kwargs)

    cli.run(bot, {})

if __name__ == "__main__":
    main()
