from __future__ import print_function
import numpy


def stat_analysis(input_lst, stride=100):
    '''
    grid input data list with customized stride
    each grid seems like left<=x<right
    '''
    max_,min_ = max(input_lst),min(input_lst)
    start = int(min_/stride)
    end = int(max_/stride) + 1
    output_lst = [0 for i in xrange(end-start)]
    for item in input_lst:
        output_lst[int(item/stride) - start] += 1
    assert sum(output_lst)==len(input_lst), 'failed: output static number does not match input'
    return output_lst, start*stride, end*stride


def write_csv(input_lst, output_file, start, end, stride=100):
    with open(output_file,'w') as f:
        f.write('axis-x,\taxis-y\n')
        for i,x in enumerate(list(numpy.arange(start, end, stride))):
            f.write('{},\t{}\n'.format(x,input_lst[i]))


if __name__ == '__main__':
    print(stat_analysis([100,200,250,300], stride=100))
