import csv
import simplejson as json


RESTAURANT_TAG = 'Restaurants'

def read_and_write_restaurants(csv_file_path, json_file_path):
    column_names = ['business_id', 'name']
    with open(csv_file_path, 'wb+') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        with open(json_file_path) as fin:
            for line in fin:
                line_contents = json.loads(line)
                # only capture restaurant data
                if RESTAURANT_TAG in line_contents.get('categories', []):
                    row = []
                    for name in column_names:
                        value = line_contents.get(name, '')
                        if isinstance(value, unicode):
                            row.append('{0}'.format(value.encode('utf-8')))
                        else:
                            row.append(value)
                    csv_file.writerow(row)


def read_and_write_users(csv_file_path, json_file_path):
    column_names = ['user_id', 'name']
    with open(csv_file_path, 'wb+') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        with open(json_file_path) as fin:
            for line in fin:
                line_contents = json.loads(line)
                row = []
                for name in column_names:
                    value = line_contents.get(name, '')
                    if isinstance(value, unicode):
                        row.append('{0}'.format(value.encode('utf-8')))
                    else:
                        row.append(value)
                csv_file.writerow(row)


def read_and_write_reviews(csv_file_path, json_file_path):
    # read restaurants.csv and contruct dictionary for restaurant ids
    restaurant_dic = {}
    with open('../../data/csv/restaurants.csv', 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            business_id = row['business_id']
            restaurant_dic[business_id] = 1


    column_names = ['business_id', 'user_id', 'stars', 'text']
    with open(csv_file_path, 'wb+') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        with open(json_file_path) as fin:
            for line in fin:
                line_contents = json.loads(line)
                row = []
                business_id = line_contents.get('business_id', '')
                # only capture restaurant reviews
                if business_id in restaurant_dic:
                    for name in column_names:
                        value = line_contents.get(name, '')
                        if isinstance(value, unicode):
                            row.append('{0}'.format(value.encode('utf-8')))
                        else:
                            row.append(value)
                    csv_file.writerow(row)

def main():
    # write restaurants to csv
    # restaurants_csv_file_path = '../../data/csv/restaurants.csv'
    # restaurants_json_file_path = '../../data/yelp_academic_dataset_business.json'
    # read_and_write_restaurants(restaurants_csv_file_path, restaurants_json_file_path)

    # write users to csv
    # users_csv_file_path = '../../data/csv/users.csv'
    # users_json_file_path = '../../data/yelp_academic_dataset_user.json'
    # read_and_write_users(users_csv_file_path, users_json_file_path)

    # write reviews to csv
    reviews_csv_file_path = '../../data/csv/reviews.csv'
    reviews_json_file_path = '../../data/yelp_academic_dataset_review.json'
    read_and_write_reviews(reviews_csv_file_path, reviews_json_file_path)


if __name__ == '__main__':
    main()