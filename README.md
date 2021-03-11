# Acerta challenge 

by Arun Chandra Sharvirala

I went with the serverless architecture to solve this challenge and chose AWS lambda for deploying my api. I will go through the process step-by-step and explain each of my files. The files are available at [GitHub](https://github.com/asharvi1/acerta).

## Project Highlights
- Multi-model API support. This model uses `Linear Regression`, `Support Vector Machine` and `Random Forest`. You can predict the house price with any of the models when you invoke the API.
- Serverless architecture.
  - `AWS Lambda`.
  - `AWS ECR` to store the docker image.
- Containerized project
  - `Docker`
- Python libraries
  - `scikit learn`
  - `joblib`
  - `numpy`
  - `pandas`

## Project Files

- `requirements.txt`
- `train_models.py`
- `app.py`
- `Dockerfile`

I will go through the files one by one:

### requirements.txt
These are the libraries I used for this project and `scikit-learn` for the machine learning models.
```text
joblib==1.0.1
numpy==1.20.1
scikit-learn==0.24.1
pandas==1.2.3
```

### train_models.py

This file loads the boston housing dataset from the `scikit-learn` library, train the machine learning models and save them as pickle files using the `joblib` library.

```python
# importing the boston housing dataset
boston_df = datasets.load_boston()

x = boston_df['data']
y = boston_df['target']
``` 

Training and saving the linear regression model as `lm.pkl`
```python
# Linear Model
linear_model_filename = 'lm.pkl'
lm = linear_model.LinearRegression()
lm.fit(x, y)

joblib.dump(lm, linear_model_filename)
```

Training and saving the support vector machine model as `svm.pkl`
```python
#Support Vector Machine
svm_filename = 'svm.pkl'
svm = svm.SVR()
svm.fit(x, y)

joblib.dump(svm, svm_filename)
```

Training and saving the random forest model as `randomfores.pkl`
```python
# Random Forest
random_forest_filename = 'randomforest.pkl'
rf_m = RandomForestRegressor()
rf_m.fit(x, y)

joblib.dump(rf_m, random_forest_filename)
```

### app.py

This file loads the contents of the request, predicts the price and returns the price predicted for the house. There are 13 independant variables in the boston housing dataset, I will assume the mean values for any of the variables that are not mentioned in the request. Final output will show the predicted price and also the list of variables that the mean values are used to predict the price.

```python
def lambda_handler(event, context):
    # get the data from the request
    body = json.loads(event['body'])

    # file names for all the models trained
    models_file_name = {
        'linear': 'lm.pkl', 
        'svm': 'svm.pkl', 
        'randomforest': 'randomforest.pkl'
        }

    mean_values = {
        'crim': 3.6135,
        'zn': 0,
        'indus': 11.3,
        'chas': 0,
        'nox': 0.5546,
        'rm': 6.28,
        'age': 68,
        'dis': 3.79,
        'rad': 9.4,
        'tax': 408,
        'ptratio': 18.45,
        'b': 356.67,
        'lstat': 12.65
    }

    input_values = []
    mean_values_used_vars = []

    # If model name is not mentioned in the request, will chose linear model to predict
    if 'model_name' in list(body.keys()):
        model_name = body['model_name']
    else:
        model_name = 'linear'

    # Checking for all variables in the request, if not will assume mean value for the not mentioned vars
    for key in list(mean_values.keys()):
        if key in list(body.keys()):
            input_values.append(body[key])
        else:
            input_values.append(mean_values[key])
            mean_values_used_vars.append(key)

    x = np.array([input_values])
    model = joblib.load(models[model_name])
    predicted_price = model.predict(x)

    if len(mean_values_used_vars) > 0:
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "Mean Values are used for the following vars": mean_values_used_vars,
                    "Predicted Price": predicted_price[0]
                }
            )
        }
    
    else:
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "Predicted price": predicted_price[0]
                }
            ),
        }

```

### Dockerfile

Loading the AWS python3.8 as the base image
```Dockerfile
FROM public.ecr.aws/lambda/python:3.8
```
Copying the files to containe
```Dockerfile
COPY app.py requirements.txt ./
COPY train_models.py ./
```
Installing the dependencies from `requirements.txt` file and training the models by running `train_models.py` file. Finally, the command to run the lambda_handler function in `app.py` file.
```Dockerfile
RUN python3.8 -m pip install -r requirements.txt -t .

RUN python3.8 train_models.py

CMD ["app.lambda_handler"]
```

# API link

`Link: `('https://v4gc7mhn1d.execute-api.us-east-1.amazonaws.com/inference')

Please find the below sample curl request to invoke the api.
```bash
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{
    "model_name": "svm",
    "crim": 0.2731,
    "zn": 0,
    "indus": 2.18,
    "chas": 0,
    "nox": 0.458,
    "rm": 6.998,
    "age": 45,
    "dis": 4.96,
    "tax": 242,
    "ptratio": 15.3,
    "b": 394.63,
    "lstat": 9.14 
}' \
  'https://v4gc7mhn1d.execute-api.us-east-1.amazonaws.com/inference'
```
