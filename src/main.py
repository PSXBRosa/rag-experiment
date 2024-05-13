from chatbot.models import QuestionAnswering
from chatbot.wrappers import AppendURLWrapper, MinimumCertaintyWrapper, AbsoluteAnswerWrapper, FetchEntireSentenceWrapper

import os
import threading
import argparse
import sqlite3
import pandas as pd
import time

# TODO -> TEST THIS FILE

def load_print(in_flag, out_flag, fps=25):
    i = 0
    states = ["   ", ".  ", ".. ", "..."]
    while not in_flag.is_set():
        print(f"[bot]: {states[i]}", end="\r")
        i = (i + 1) % 4
        time.sleep(1/fps)
    out_flag.set()

def main():
    parser = argparse.ArgumentParser(prog='main')
    parser.add_argument("-f", "--context-file")
    parser.add_argument("-k", "--k-docs", default=3)
    argv = parser.parse_args()

    context_file = argv.context_file
    if context_file is None:
        parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        context_file = os.path.join(parent_directory, "assistance.sqlite3")

    cnx = sqlite3.connect(context_file)
    df = pd.read_sql_query("SELECT * FROM assistance", cnx)

    wrappers = [
        (AppendURLWrapper, [], {}),
        (MinimumCertaintyWrapper, [0.05], {}),
        (AbsoluteAnswerWrapper, [], {}),
    ]

    interpreter = QuestionAnswering(df)
    for wrapper, args, kwargs in wrappers:
        interpreter = wrapper(interpreter, *args, **kwargs)

    def process_query(in_flag, out_flag, query):
        ans = interpreter.answer(query, argv.k_docs)
        out_flag.set()
        in_flag.wait()
        print(f"[bot]: {ans}")

    print("[bot]: Bonjour, votre assistant a été initialisée. Tappez 'ctrl + c' pour quitter l'app. Vous pouvez me poser n'importe quelle question et j'essaierai d'y répondre du mieux que je peux !")

    try:
        while True:
            query = input(">>> ")

            computation_done = threading.Event()
            printer_done = threading.Event()

            print_thread = threading.Thread(target=load_print, args=(computation_done, printer_done, 4))
            query_thread = threading.Thread(target=process_query, args=(printer_done, computation_done, query))

            print_thread.start()
            query_thread.start()

            query_thread.join()
            print_thread.join()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()