#! /usr/bin/env python

import json

data_dir = './../data/'

raw_business_file = data_dir + 'yelp_academic_dataset_business.json'
raw_review_file = data_dir + 'yelp_academic_dataset_review.json'

def get_restaruants():
    restaurant_list = []

    with open(raw_business_file, 'r') as f:
        while True:
            line = f.readline()

            if not line:
                break

            business_json = json.loads(line)
            business_category = business_json['categories']

            if 'Restaurants' in business_category:
                restaurant_list.append(business_json['business_id'])
                
    print(len(restaurant_list))

    return restaurant_list

def get_reviews(restaurant_list):
    with open(data_dir + 'reviews.txt', 'w') as fw:

        with open(raw_review_file, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                review_json = json.loads(line)            

                if review_json['business_id'] in restaurant_list:
                    fw.write(review_json['text'])
                    
    
def main():
    restaurant_list = get_restaruants()

    get_reviews(restaurant_list)

if __name__ == '__main__':
    main()