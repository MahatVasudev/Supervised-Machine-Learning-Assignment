import argparse


def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "y")


def make_datalist(v: str):
    v_list = v.split(',')
    v_list[0] = v_list[0].removeprefix("[").removeprefix("(").strip()
    v_list[-1] = v_list[-1].removesuffix("]").removesuffix(")").strip()

    return v_list


# NOTE: For ML scripts
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int)
parser.add_argument("--filename")
parser.add_argument('--verbose', type=str2bool)
parser.add_argument('--log-loss', type=str2bool)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=0.0005)
parser.add_argument('--scheduler-patience', type=int, default=3)


# NOTE: For data_scripts
download_script_parser = argparse.ArgumentParser()
download_script_parser.add_argument("--years", type=make_datalist)
download_script_parser.add_argument("--mode", type=int, choices=[0, 1, 2])
download_script_parser.add_argument("--batch_size", type=int)
download_script_parser.add_argument("--bin_size", type=float, default=0.25)
download_script_parser.add_argument("--time", type=str, default="1d")
