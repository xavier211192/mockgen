import pandas as pd
from faker import Faker
import uuid
from datetime import datetime, timedelta

fake = Faker()

def generate_data():
    data = {
        'user_id': [str(uuid.uuid4()) for _ in range(25)],
        'full_name': [fake.name() for _ in range(25)],
        'email': [fake.email(name=name) for name in [fake.name() for _ in range(25)]],
        'signup_date': [datetime(2023, 1, i+1) for i in range(25)],
        'country': [fake.country() for _ in range(25)],
        'is_active': [fake.pybool() for _ in range(25)]
    }
    
    # Ensure is_active is 70% True
    for i in range(len(data['is_active'])):
        if fake.pybool() and data['is_active'][i] == False:
            data['is_active'][i] = True
        elif not fake.pybool() and data['is_active'][i] == True:
            data['is_active'][i] = False
            
    df = pd.DataFrame(data)
    return df

df = generate_data()
df.to_csv('output.csv', index=False)