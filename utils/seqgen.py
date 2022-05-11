import argparse
import random

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--length', metavar='<size>', required=True,
                    type=int, help='length of amino-acid sequences')
parser.add_argument('-s', '--sample', metavar='<count>', required=True,
                    type=int, help='count of sequences for each length')
parser.add_argument('-o', '--output', metavar='<file>', required=True,
                    type=str, help='file where sequences stored')

args = vars(parser.parse_args())

sequence_length = args['length']
sample_count = args['sample']
output = open(args['output'], 'a')

amino_acids = "acdefghiklmnpqrstwyv"

for _ in range(sample_count):
    src = ''.join(random.choice(amino_acids) for _ in range(sequence_length))
    ref = ''.join(random.choice(amino_acids) for _ in range(sequence_length))

    output.write(src + ' ' + ref + '\n')

print("Sequences have been generated!")
output.close()
