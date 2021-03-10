FROM public.ecr.aws/lambda/python:3.8

# copying the files to container
COPY app.py requirements.txt ./
COPY train_models.py ./

# installing the dependencies 
RUN python3.8 -m pip install -r requirements.txt -t .

# training the models
RUN python3.8 train_models.py

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
