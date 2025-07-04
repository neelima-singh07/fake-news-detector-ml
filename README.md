# 📰 Fake News Detector

A machine learning-powered web application built with Streamlit to detect fake news articles. This project  analyze news content and predict whether it's real or fake.

## Deployed Link : (https://fake-news-detector-ml-yfir23yax6pxmlf7irpp73.streamlit.app/)
## 🚀 Features

- **Real-time Analysis**: Instantly analyze news articles for authenticity
- **Confidence Scoring**: Get confidence levels for predictions
- **Text Insights**: View detailed statistics about the text
- **Abusive Content Detection**: Identify potentially harmful language
- **User-friendly Interface**: Clean and intuitive Streamlit dashboard

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation
- **Pickle** - Model serialization

## 📁 Project Structure

```
Fake-news-detection/
│
├── app.py              # Main Streamlit application
├── train_model.py      # Model training script
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore file
├── README.md          # Project documentation
│
├── data/              # Dataset folder
│   ├── Fake.csv       # Fake news dataset
│   └── True.csv       # Real news dataset
│
└── model/             # Trained models
    ├── model.pkl      # Trained classifier
    └── vectorizer.pkl # Text vectorizer
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/Fake-news-detection.git
cd Fake-news-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501`

## 🎯 How to Use

1. **Enter News Text**: Paste or type the news article you want to analyze
2. **Click Analyze**: Press the "🔍 Analyze News" button
3. **View Results**: See the prediction, confidence score, and detailed insights
4. **Check Warnings**: Review any detected abusive content or borderline cases

## 📊 Model Performance

The model is trained on a dataset of real and fake news articles and provides:
- Binary classification (Real/Fake)
- Confidence scoring
- Text analysis metrics


## 🔮 Future Enhancements

- [ ] Add more sophisticated NLP models
- [ ] Implement source credibility checking
- [ ] Add multilingual support
- [ ] Include news category classification
- [ ] Deploy to cloud platforms

---


