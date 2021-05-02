import uvicorn
from fastapi import FastAPI
from FakeNewss import FakeNews
from RelatedNewss import RelatedNews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)
tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
df = pd.read_csv('news.csv')
x = df['text']
y = df['label']
df['label'].dropna(inplace=True)
df['text'].dropna(inplace=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(News_text):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_text = [News_text]
    vectorized_input_text = tfvect.transform(input_text)
    result = classifier.predict(vectorized_input_text)
    return result

@app.post('/predict')
def predict_fake_or_real(data:FakeNews):
    data = data.dict()
    News_text=data['News_text']
    
    prediction = fake_news_det(News_text)
    if(prediction == ['FAKE']):
        prediction="⚠️News is fake!"
        ob=RelatedNews()
        return ob.related()
    elif(prediction == ['TRUE']):
        prediction="📰News is real!"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)