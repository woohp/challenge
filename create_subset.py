import sys
import json
import random
from collections import defaultdict


def main(args):
    business_file = open('yelp_academic_dataset_business.json')
    user_file = open('yelp_academic_dataset_user.json')
    review_file = open('yelp_academic_dataset_review.json')

    businesses = []
    users = []
    reviews = []

    businesses = [json.loads(line) for line in business_file]

    users = [json.loads(line) for line in user_file]

    reviews1 = defaultdict(list) # key is user_id
    reviews2 = defaultdict(list) # key is business_id
    for line in review_file:
        review = json.loads(line)
        reviews1[review['user_id']].append(review)
        reviews2[review['business_id']].append(review)


    num_users = 50
    num_businesses = 50

    users_subset = []
    businesses_subset = []
    reviews_subset = []
    random.seed(5)
    while 1:
        if len(users_subset) < num_users:
            while 1:
                if len(users_subset) == 0 or random.random() < len(users_subset) / num_users * 10:
                    new_user = random.sample(users, 1)[0]
                else:
                    b = random.sample(businesses_subset, 1)[0]
                    new_user = random.sample(reviews2[b], 1)[0]
                new_user_id = new_user['user_id']
                if new_user_id not in users_subset: break
            users_subset.append(new_user_id)

        if len(businesses_subset) < num_businesses:
            while 1:
                if len(businesses_subset) == 0 or random.random() < len(businesses_subset) / num_businesses * 10:
                    new_business = random.sample(businesses, 1)[0]
                else:
                    u = random.sample(users_subset, 1)[0]
                    new_business = random.sample(reviews1[u], 1)[0]
                new_business_id = new_business['business_id']
                if new_business_id not in businesses_subset: break
            businesses_subset.append(new_business_id)

        if len(users_subset) == num_users and len(businesses_subset) == num_businesses:
            break

    user_id_indices = {}
    for i, user_id in enumerate(users_subset):
        user_id_indices[user_id] = i
    business_id_indices = {}
    for i, business_id in enumerate(businesses_subset):
        business_id_indices[business_id] = i

    reviews_subset_matrix = [[0 for i in xrange(num_users)] for j in xrange(num_businesses)]

    for user_id in users_subset:
        for review in reviews1[user_id]:
            business_id = review['business_id']
            if business_id in businesses_subset:
                reviews_subset.append((user_id, business_id, review['stars']))

                business_id_index = business_id_indices[business_id]
                user_id_index = user_id_indices[user_id]
                reviews_subset_matrix[business_id_index][user_id_index] = review['stars']

    for user_id in users_subset:
        print user_id
    print
    for business_id in businesses_subset:
        print business_id
    print
    for user_id, business_id, stars in reviews_subset:
        print user_id, business_id, stars

#    print len(set(users_subset))
#    print len(set(businesses_subset))
#    print len(set(reviews_subset))
#    for row in reviews_subset_matrix:
#        print row
    

if __name__ == '__main__':
    main(sys.argv)
