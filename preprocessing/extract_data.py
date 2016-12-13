import csv


def read_users_data_into_dic(read_num):
    path = '../../data/csv/users.csv'

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

def read_restaurants_data_into_dic(read_num):
    path = '../../data/csv/restaurants.csv'

    restaurant_dic = {}
    row_count = 0
    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if row_count >= read_num:
                break

            restaurant_id = row['business_id']
            restaurant_dic[restaurant_id] = row_count
            row_count += 1
    return restaurant_dic


def process_reviews_data(user_dic, restaurant_dic, ratings_out_path, reviews_out_path):
    in_path = '../../data/csv/reviews.csv'
    rating_column_names = ['user_id', 'restaurant_id', 'rating']
    review_column_names = ['user_id', 'restaurant_id', 'text']
    # used for sorting
    user_map = {}
    user_set = set()
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
                    key = user_dic[user_id]
                    value = [restaurant_dic[restaurant_id], stars, review_text]
                    if key not in user_map:
                        user_map[key] = []

                    user_map[key].append(value)
                    user_set.add(key)
        # sort the user
        user_set = sorted(user_set)
        for user_id in user_set:
            for line in user_map[user_id]:
                rating_row = [user_id, line[0], line[1]]
                review_row = [user_id, line[0], line[2]]
                csv_file_rating.writerow(rating_row)
                csv_file_review.writerow(review_row)


def main():
    data_num = 10000   

    # separate rating and review data
    user_dic = read_users_data_into_dic(data_num)
    restaurant_dic = read_restaurants_data_into_dic(data_num)
    ratings_out_path = '../../data/preprocessed_data/ratings.csv'
    reviews_out_path = '../../data/preprocessed_data/reviews.csv'
    process_reviews_data(user_dic, restaurant_dic, ratings_out_path, reviews_out_path)



if __name__ == '__main__':
    main()