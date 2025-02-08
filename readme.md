# Task 1: Generate business attributes

Things like "Japanese" or "Cinema" or "Sake" should be extracted

# Task 2: Generate a list, per business, of negative, improvable aspects

Things such as "dirty bathrooms" should be extracted from reviews.


## Kaggle data set

Download only the following three ndjson files:

* yelp_academic_dataset_business.json
* yelp_academic_dataset_review.json
* yelp_academic_dataset_tip.json

save them into a folder named `data` at the root directory

## Python environment

Create a venv in the root folder:

```shell
python -m venv .venv
```

Install the required packages by running 

```shell
pip install -r requirements.txt
```


### Notes for running the `clean_data.py` file

Running this file requires a decent amount of memory, greater than or equal to 16gb. 