import sys
import wundy
from wundy import ui
from wundy import first

def run():
    p = argparse.ArgumentParser()
    p.add_argument("file", help = "Wundy yaml file")
    args = p.parse_args()
    with open(args.file) as fh:
        data = ui.load(fh)
    inp = ui.preprocess(data)
    soln = wundy.first.first_fe_code(
        inp["coords"],
        inp["blocks"],
        inp["bcs"],
        inp["dload"],
        inp["materials"],
        inp["block_elem_map"],
    )