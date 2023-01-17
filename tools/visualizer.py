import argparse
import matplotlib.pyplot as plt


class Log:
    def __init__(self, name, file):
        self.name = name
        self.file = file

        self.data = dict()
        self.aver = dict()

    def parse(self):
        with open(self.file) as file:
            self.__read(file)

        for length, time in self.data.items():
            self.aver[length] = round(sum(time) / len(time))

    def __read(self, file):
        for line in file:
            length, time, _ = line.strip().split(',')
            self.data.setdefault(int(length), list()).append(int(time))


class Plotter:
    def __init__(self):
        self.__log = list()

    def add_log(self, log):
        self.__log.append(log)

    def plot(self):
        self.add_line_graph()
        plt.show()

    def add_line_graph(self):
        for log in self.__log:
            x = list(log.aver.keys())
            y = list(log.aver.values())

            plt.rcParams['toolbar'] = 'None'
            plt.figure("Line Graph")

            plt.plot(x, y, label=log.name)

        plt.xlabel('Sequence Length')
        plt.ylabel('Execution Time(ms)')

        plt.legend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs=2, metavar=('<name>', '<file>'),
                        type=str, action='append', required=True,
                        help='adds log file to be visualized')

    args = vars(parser.parse_args())
    logs = args['input']

    plotter = Plotter()

    for name, file in logs:
        log = Log(name, file)
        log.parse()

        plotter.add_log(log)

    plotter.plot()
