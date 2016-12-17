import csv


def read_users_data_into_dic(read_num, path):
    user_dic = {}
    row_count = 0
    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if row_count >= read_num:
                break

            user_id = row['user_id']
            user_dic[user_id] = row_count
            row_count += 1
    return user_dic


def modified_read_users_data_into_dic():
    user_path = '../../data/csv/users.csv'

    user_dic = {}
    with open(user_path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            user_id = row['user_id']
            user_dic[user_id] = False

    # read reviews
    review_path = '../../data/csv/reviews.csv'
    with open(review_path, 'rb') as fin:
        reader = csv.DictReader(fin, delimiter=',')
        for row in reader:
            user_id = row['user_id']
            if user_id in user_dic:
                user_dic[user_id] = True

    return user_dic

def rewrite_users(user_dic, original_user_path, rewrite_user_path):
    column_names = ['user_id', 'name']
    with open(rewrite_user_path, 'wb+') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(column_names)
        with open(original_user_path, 'rb') as fin:
            reader = csv.DictReader(fin, delimiter=',')
            for row in reader:
                user_id = row['user_id']
                if user_dic.get(user_id, False):
                    csv_file.writerow([row['user_id'], row['name']])



def read_restaurants_data_into_dic():
    """
    read all restaurant id
    """
    path = '../../data/csv/restaurants.csv'

    restaurant_dic = {}
    row_count = 0
    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            restaurant_id = row['business_id']
            restaurant_dic[restaurant_id] = row_count
            row_count += 1
    return restaurant_dic


def process_reviews_data(user_dic, restaurant_dic, ratings_out_path, reviews_out_path):
    test_dic = {}
    for user_id in user_dic:
        test_dic[user_id] = False


    in_path = '../../data/csv/reviews.csv'
    rating_column_names = ['user_id', 'restaurant_id', 'rating']
    review_column_names = ['user_id', 'restaurant_id', 'text']
    # used for sorting
    user_map = {}
    user_set = set()
    restaurant_set = set()
    with open(ratings_out_path, 'wb+') as fout_rating, open(reviews_out_path, 'wb+') as fout_review:
        csv_file_rating = csv.writer(fout_rating)
        csv_file_review = csv.writer(fout_review)

        csv_file_rating.writerow(list(rating_column_names))
        csv_file_review.writerow(list(review_column_names))
        with open(in_path, 'rb') as fin:
            reader = csv.DictReader(fin, delimiter=',')
            for row in reader:
                user_id = row['user_id']
                restaurant_id = row['business_id']
                stars = row['stars']
                review_text = row['text']
                if user_id in user_dic and restaurant_id in restaurant_dic:
                    # test
                    test_dic[user_id] = True

                    key = user_dic[user_id]
                    value = [restaurant_dic[restaurant_id], stars, review_text]
                    if key not in user_map:
                        user_map[key] = []

                    user_map[key].append(value)
                    user_set.add(key)
                    restaurant_set.add(restaurant_id)
                # else:
                #     if user_id not in user_dic:
                #         print 'what the fuck: ', user_id
        
        print 'user_num: ', len(user_set)
        print 'restaurant_num: ', len(restaurant_set)

        #test
        for user_id in test_dic:
            if not test_dic[user_id]:
                print 'omg: ', user_id

        # sort the user
        user_set = sorted(user_set)
        for user_id in user_set:
            for line in user_map[user_id]:
                rating_row = [user_id, line[0], line[1]]
                review_row = [user_id, line[0], line[2]]
                csv_file_rating.writerow(rating_row)
                csv_file_review.writerow(review_row)


def main():
    # user_num = 5000
    user_num = 478841
    # user_num = 0

    # delete user that has no reviews
    # original_user_path = '../../data/csv/users.csv'
    # rewrite_user_path = '../../data/csv/rewrite_users.csv'
    # user_dic = modified_read_users_data_into_dic()
    # rewrite_users(user_dic, original_user_path, rewrite_user_path)

    # separate rating and review data
    path = '../../data/csv/rewrite_users.csv'
    
    user_dic = read_users_data_into_dic(user_num, path)
    restaurant_dic = read_restaurants_data_into_dic()
    ratings_out_path = '../../data/preprocessed_data/ratings.csv'
    reviews_out_path = '../../data/preprocessed_data/reviews.csv'
    process_reviews_data(user_dic, restaurant_dic, ratings_out_path, reviews_out_path)



if __name__ == '__main__':
    main()