import time
import threading


def process_query(in_flag, out_flag, bot, query):
    """send the query to be processed"""
    ans = bot.answer(query)
    out_flag.set()
    in_flag.wait()
    print(f"[bot]: {ans}")


def load_print(in_flag, out_flag, fps=25):
    """print a loading animation while a flag is not set"""
    i = 0
    states = ["   ", ".  ", ".. ", "..."]
    while not in_flag.is_set():
        print(f"[bot]: {states[i]}", end="\r")
        i = (i + 1) % 4
        time.sleep(1 / fps)
    out_flag.set()


def run(bot, argv={}):
    """run the cli"""
    print(
        "[bot]: Bonjour, votre assistant a été initialisée. Tappez 'ctrl + c' pour quitter l'app. "
        "Vous pouvez me poser n'importe quelle question et j'essaierai d'y répondre du mieux que je peux !"
    )

    try:
        while True:
            query = input(">>> ")

            computation_done = threading.Event()
            printer_done = threading.Event()

            print_thread = threading.Thread(
                target=load_print, args=(computation_done, printer_done, 4)
            )
            query_thread = threading.Thread(
                target=process_query,
                args=(printer_done, computation_done, bot, query),
            )

            print_thread.start()
            query_thread.start()

            query_thread.join()
            print_thread.join()
    except KeyboardInterrupt:
        pass
