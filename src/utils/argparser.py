import argparse


def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "y")


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int)
parser.add_argument("--filename")
parser.add_argument('--verbose', type=str2bool)
parser.add_argument('--log-loss', type=str2bool)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=0.0005)
parser.add_argument('--scheduler-patience', type=int, default=3)
