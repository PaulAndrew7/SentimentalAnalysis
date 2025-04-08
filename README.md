# Sentiment Analysis Web Application

This is a simple web application that performs sentiment analysis on text reviews using machine learning models.

## Project Structure
```
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── templates/            # Frontend files
│   └── index.html        # Main web interface
├── models/              # Directory for your trained models
│   ├── model.pkl        # Your trained model (to be added)
│   └── vectorizer.pkl   # Your text vectorizer (to be added)
```

## Setup Instructions

1. **Convert your Jupyter Notebook**
   - Export your model and vectorizer from your notebook:
   ```python
   # At the end of your notebook, add:
   import pickle
   
   # Save the model
   with open('models/model.pkl', 'wb') as f:
       pickle.dump(model, f)
   
   # Save the vectorizer
   with open('models/vectorizer.pkl', 'wb') as f:
       pickle.dump(vectorizer, f)
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Access the Web Interface**
   - Open your browser and go to `http://localhost:5000`
   - Enter text in the input field and click "Analyze Sentiment"

## Notes
- Make sure to create a 'models' directory and save your trained model and vectorizer there
- The application expects your model to predict binary sentiment (positive/negative)
- Adjust the model loading code in `app.py` if your model implementation differs