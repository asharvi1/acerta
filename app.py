import json
import numpy as np
import joblib

def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

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
