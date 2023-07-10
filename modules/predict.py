import json
import os
from datetime import datetime

import dill
import pandas as pd
from pydantic import BaseModel

test_dir = 'data/test'
predict_dir = 'data/predictions'

def predict():
    with open('data/models/cars_pipe_202307081857.pkl', 'rb') as f:
        model = dill.load(f)
    # print(type(model.named_steps["classifier"]).__name__)

    tests = pd.DataFrame()

    for path in os.listdir(test_dir):
        file_path = os.path.join(test_dir, path)
        if os.path.isfile(file_path):
            with open(file_path, '+r') as file:
                test = json.load(file)
                tests = pd.concat([tests, pd.DataFrame.from_dict([test])], axis=0)
    
    tests['y'] = model.predict(tests)
    tests[['id', 'y']].to_csv(os.path.join(predict_dir, f'predict_{datetime.now().strftime("%Y%m%d%H%M")}.csv'),
                              index=False)
        



if __name__ == '__main__':
    predict()
