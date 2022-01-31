import argparse
import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline


class Parser:
    def __init__(self, name, file):
        self.name = name
        self.__file = file

        self.data = dict()
        self.aver = dict()

    def parse(self):
        with open(self.__file) as file:
            self.__read(file)

        next(iter(self.data.values())).pop(0)
        self.__average()

    def __read(self, file):
        for line in file:
            length, _, time = line.strip().split(',')
            self.data.setdefault(int(length), list()).append(int(time))

    def __average(self):
        for length, time in self.data.items():
            self.aver[length] = round(sum(time) / len(time))


class Plotter:
    def __init__(self):
        self.__log = list()

    def add_log(self, log):
        self.__log.append(log)

    def plot(self):
        plt.rcParams['toolbar'] = 'None'

        for log in self.__log:
            log.parse()

            x = list(log.aver.keys())
            y = list(log.aver.values())

            spline = make_interp_spline(x, y)

            x = np.linspace(min(x), max(x), len(x) * 10)
            y = spline(x)

            plt.plot(x, y, label=log.name)

        plt.xlabel('Sequence Length')
        plt.ylabel('Execution Time(ms)')

        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs=2, metavar=('<name>', '<file>'),
                        type=str, action='append', required=True,
                        help='adds log file to be visualized')

    args = vars(parser.parse_args())
    logs = args['input']

    plotter = Plotter()

    for name, file in logs:
        plotter.add_log(Parser(name, file))

    plotter.plot()
