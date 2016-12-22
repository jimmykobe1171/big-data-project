# Multi-aspect Sentiment Analysis and Recommender Systems       

This project build a recommander system using collaborative filtering with multi-aspect sentiment analysis. There are two folder in the project directory.

* preprocessing. This folder includes the data preprocessing scripts. For example, "json_to_csv.py" convert the original json data file to csv format.

* recommandation. This folder includes the codes for building the recommandation model. For example, "col_filter.py" is the main file for collaborative filtering model trainning and testing.

## How to run
To run the recommander system, you need to move in the recommandation directory, and then run the python script "col_filter.py" like this:

```
python col_filter.py
```

Then the script will run the training and testing process of the recommander system and output the error rate. The two .p files are the pickled feature vectors for users and restaurants respectively.
