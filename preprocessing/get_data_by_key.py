#! /usr/bin/env python

import json

raw_business_file = 'yelp_academic_dataset_business.json'
raw_user_file = 'yelp_academic_dataset_user.json'
raw_review_file = 'yelp_academic_dataset_review.json'
raw_tip_file = 'yelp_academic_dataset_tip.json'
raw_checkin_file = 'yelp_academic_dataset_checkin.json'

def get_json(file_name):
    with open(file_name, 'r') as json_file:
        lines = json_file.readlines()

    return [json.loads(s) for s in lines]

def get_restaruants():
    business_json_list = get_json(raw_business_file)

    business_category_set = set([])
    restaurant_list = []
    for business_json in business_json_list:
        business_category = business_json['categories']

        if 'Restaurants' in business_category:
            for category in business_category:
                business_category_set.add(category)
            restaurant_list.append(business_json['business_id'])
            # print(business_category)

    print(len(business_category_set))
    print(len(business_json_list))
    print(len(restaurant_list))

    return restaurant_list

def get_reviews(restaurant_list):
    review_json_list = get_json('review_100.json')

    print len(review_json_list)

    restaurant_reviews = []

    for review_json in review_json_list:
        if review_json['business_id'] in restaurant_list:
            restaurant_reviews.append(review_json)

    print len(restaurant_reviews)

def main():
    restaurant_list = get_restaruants()

    get_reviews(restaurant_list)

if __name__ == '__main__':
    main()