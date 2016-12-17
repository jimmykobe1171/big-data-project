import json

data_dir = './../data/'

raw_business_file = data_dir + 'yelp_academic_dataset_business.json'
raw_review_file = data_dir + 'yelp_academic_dataset_review.json'

def get_restaruants():
    restaurant_list = []

    with open(raw_business_file, 'r', encoding='utf-8', errors='ignore') as f:
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
    with open(data_dir + 'reviews_utf_8.txt', 'w', encoding='utf-8', errors='ignore') as fw:

        with open(raw_review_file, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                review_json = json.loads(line)            

                if review_json['business_id'] in restaurant_list:
                    fw.write(review_json['text'])

def get_users(restaurant_list):
    user_num = 0
    review_num = 0
    user_id_set = set()

    with open(raw_review_file, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            line = f.readline()
            if not line:
                break

            review_json = json.loads(line)            

            if review_json['business_id'] in restaurant_list:

                review_num += 1

                user_id = review_json['user_id']

                if user_id not in user_id_set:
                    user_id_set.add(user_id)

                    user_num += 1

    print(user_num)
    print(review_num)


def get_restaurant(restaurant_list):
    with open(data_dir + 'restaurant_reviews.txt', 'w', encoding='utf-8', errors='ignore') as fw:
        with open(raw_review_file, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                review_json = json.loads(line)            

                if review_json['business_id'] in restaurant_list:
                    fw.write(line)

def get_users_restaurant(restaurant_list):

    user_id_list = []

    with open(data_dir + 'restaurant_reviews.txt', 'w', encoding='utf-8', errors='ignore') as fw:
        with open(raw_review_file, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                review_json = json.loads(line)            

                if review_json['business_id'] in restaurant_list:
                    fw.write(line)

                    user_id = review_json['user_id']

                    if user_id not in user_id_list:
                        user_id_list.append(user_id)

    print(len(user_id_list))

    user_id_list.sort()
    restaurant_list.sort()

    with open(data_dir + 'user_list.txt', 'w', encoding='utf-8', errors='ignore') as fw:
        for user_id in user_id_list:
            fw.write(user_id.strip() + '\n')

    with open(data_dir + 'restaurant_list.txt', 'w', encoding='utf-8', errors='ignore') as fw:
        for r_id in restaurant_list:
            fw.write(r_id.strip() + '\n')


        
def main():
    restaurant_list = get_restaruants()

    # get_reviews(restaurant_list)

    # get_users(restaurant_list)

    # get_restaruants(restaurant_list)

    get_users_restaurant(restaurant_list)

if __name__ == '__main__':
    main()



