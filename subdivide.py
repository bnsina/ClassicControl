import argparse
import math

def buncher(lbd, ubd, ns, flag):
    res = []
    scale = (ubd-lbd) / math.log(math.log(1.0 + ns))

    left = (not flag)*ubd + flag*lbd

    for i in range(ns):
    
        right = (not flag)*ubd + flag*lbd + ((-1)**(not flag)) * math.log(math.log(3.0 + i)) * scale
        res.append(left)
        left = right
    
    return sorted(res)

def displayer(lst, flag):
    
    if flag == True:
        print(lst)
    else:
        for i in reversed(range(len(lst))):
            print(f'{lst[i]} \n')
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='Subdivision of interval',
        description='DEFAULT = loglog'
    )
    parser.add_argument('lower_bd', nargs=1, type=float)
    parser.add_argument('upper_bd', nargs=1, type=float)
    parser.add_argument('num_subs', nargs=1, type=int)
    parser.add_argument('-r', '--right', dest='right', action='store_true', help='right bunched')
    parser.add_argument('-l', '--list', dest='list', action='store_true', help='print as list')
    args = parser.parse_args()
    
    out = buncher(args.lower_bd[0], args.upper_bd[0], args.num_subs[0], args.right)
    
    displayer(out, args.list)
    