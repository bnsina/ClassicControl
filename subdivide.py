import argparse
import math

def left_bunch(lbd, ubd, ns):
    scale = (ubd-lbd) / math.log(1.0 + ns)
    left = ubd

    for i in range(ns):
        right = ubd - math.log(2.0 + i) * scale
        print(f'{left} \n')
        left = right

def right_bunch(lbd, ubd, ns):
    scale = (ubd-lbd) / math.log(1.0 + ns)
    left = lbd

    for i in range(ns):
        right = lbd + math.log(2.0 + i) * scale
        print(f'{left} \n')
        left = right



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='Logarithmic subdivision of interval',
        description='DEFAULT = exponential'
    )
    parser.add_argument('lower_bd', nargs=1, type=float)
    parser.add_argument('upper_bd', nargs=1, type=float)
    parser.add_argument('num_subs', nargs=1, type=int)
    parser.add_argument('-r', '--right', dest='r', action='store_true', help='right bunched')
    args = parser.parse_args()
    
    if args.r == False:
        left_bunch(args.lower_bd[0], args.upper_bd[0], args.num_subs[0])
    else:
        right_bunch(args.lower_bd[0], args.upper_bd[0], args.num_subs[0])
    