#!/usr/bin/env python3
import pickle
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)

    args = parser.parse_args()

    obj = pickle.load(open(args.file, 'rb'))

    if isinstance(obj, plt.Figure):
        obj.show()
        input()
    elif isinstance(obj, list):
        rslt = []
        for idx, item in enumerate(obj):
            if item < -10000:
                pass
            else:
                rslt.append(item)
        fig = plt.figure()
        plt.plot(range(len(rslt)), rslt)
        plt.show()
        input()
    pass


