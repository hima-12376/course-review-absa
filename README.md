# Course Review Topic and Aspect-Based Sentiment Analyzer

## Overview

The Course Review Topic and Aspect-Based Sentiment Analyzer is a Natural Language Processing (NLP) application that analyzes student course reviews to identify important course aspects and determine the sentiment associated with each aspect. The system extracts topics such as instructors, assignments, examinations, and syllabus, enabling detailed analysis beyond overall review sentiment.

The project combines traditional NLP techniques with transformer-based deep learning models to provide accurate aspect-level sentiment classification through an interactive web application.

---

## Features

- Extracts course-related aspects from student reviews using spaCy.
- Performs Aspect-Based Sentiment Analysis (ABSA) at the aspect level.
- Utilizes a hybrid transformer architecture combining DeBERTa and RoBERTa models.
- Employs ensemble probability fusion for improved sentiment prediction.
- Classifies sentiments into Positive, Neutral, and Negative categories.
- Provides confidence scores for each prediction.
- Interactive Flask web application for real-time review analysis.
- Dashboard for visualizing sentiment distribution and analytical insights.

---

## Project Architecture

```
Student Reviews
        │
        ▼
Data Preprocessing
        │
        ▼
Aspect Extraction (spaCy)
        │
        ▼
Hybrid Transformer Models
(DeBERTa + RoBERTa)
        │
        ▼
Ensemble Probability Fusion
        │
        ▼
Sentiment Prediction
        │
        ▼
Flask Web Application
        │
        ▼
Interactive Dashboard
```

---

## Tech Stack

### Programming Language

- Python

### NLP & Machine Learning

- spaCy
- Hugging Face Transformers
- DeBERTa
- RoBERTa
- Scikit-learn

### Web Framework

- Flask

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Chart.js

---

## Dataset

The project uses a dataset containing student course reviews collected from online learning platforms. Reviews are processed to extract meaningful course aspects and their corresponding sentiment.

Example aspects include:

- Instructor
- Assignments
- Examinations
- Course Content
- Syllabus

---

## Installation

Clone the repository:

```bash
git clone https://github.com/hima-12376/course-review-absa.git
```

Navigate to the project directory:

```bash
cd course-review-absa
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the Flask application:

```bash
python main.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

Enter a course review to analyze the extracted aspects and corresponding sentiment predictions.

---

## Results

The system is capable of:

- Extracting multiple aspects from a single review.
- Predicting aspect-level sentiment using transformer-based models.
- Displaying confidence scores for each sentiment prediction.
- Presenting interactive visualizations through a web dashboard.

---

## Project Structure

```
course-review-absa/
│
├── data/
├── static/
├── templates/
├── aspect_extractor.py
├── sentiment_model.py
├── topic_model.py
├── train_absa.py
├── train_roberta.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Support additional transformer architectures such as BERT and DistilBERT.
- Deploy the application using Docker and cloud platforms.
- Integrate multilingual sentiment analysis.
- Improve aspect extraction using dependency parsing.
- Extend the dashboard with advanced analytics and trend visualization.
- Enable batch review analysis through file upload functionality.

---

## Author

**Hima Paul**

Final Year B.Tech Computer Science Engineering Student

Interested in Artificial Intelligence, Machine Learning, Computer Vision, and Natural Language Processing.

---
