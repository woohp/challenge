import sys
import random
import scipy as sp
import scipy.sparse as sparse

def main(args):
    # parse the input file
    subset_file = open('subset.txt')
    num_empty_lines = 0
    users = []
    businesses = []
    reviews = [] # tuples of (user_id, business_id, stars)
    for line in subset_file:
        if len(line) == 0:
            num_empty_lines += 1
        else:
            if num_empty_lines == 0:
                users.append(line.strip())
            elif num_empty_lines == 1:
                businesses.append(line.strim())
            else:
                review = line.split()
                review[2] = int(int(review[2]) > 3)

    num_attributes = 1
    # vectors of attributes relevant to the worker
    x = [[random.random() for j in xrange(num_attributes)] for i in xrange(len(businesses))] 
    # vectors of strengths of each worker
    w = [[random.random() for j in xrange(num_attributes)] for i in xrange(len(users))] 
    # bias of each worker
    tau = [random.random() for i in xrange(len(users))]
    # noise of each worker
    sigma = [random.random() for i in xrange(len(users))]



    

if __name__ == '__main__':
    main(sys.argv)
