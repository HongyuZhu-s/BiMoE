from Pre_processing import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='DEAP')
    parser.add_argument('--data-path', type=str, default='deap_data')
    parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='L', choices=['A', 'V','D','L'])
    parser.add_argument('--segment', type=int, default=4, help='segment length in seconds')
    parser.add_argument('--trial-duration', type=int, default=60, help='trial duration in seconds')
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=128)
    parser.add_argument('--input-shape', type=tuple, default=(1, 28, 512))
    parser.add_argument('--data-format', type=str, default='raw')

    parser.add_argument('--T', type=int, default=15)

    parser.add_argument('--hidden', type=int, default=32)

    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()

    sub_to_run = np.arange(args.subjects)

    pd = PrepareData(args)
    pd.run(sub_to_run, split=True, feature=False)



