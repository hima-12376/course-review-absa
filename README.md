# Course Review Topic and Aspect-Based Sentiment Analyzer

## Overview

The Course Review Topic and Aspect-Based Sentiment Analyzer is a Natural Language Processing (NLP) application that performs Aspect-Based Sentiment Analysis (ABSA) on student course reviews. Instead of predicting only the overall sentiment of a review, the system identifies important course aspects such as instructors, assignments, examinations, and syllabus, and determines the sentiment expressed toward each aspect.

The application combines aspect extraction using spaCy with transformer-based sentiment classification and provides the results through an interactive Flask web interface.

---

## Features

- Extracts course-related aspects from student reviews using spaCy.
- Performs Aspect-Based Sentiment Analysis (ABSA).
- Uses transformer-based models (DeBERTa and RoBERTa) for sentiment prediction.
- Classifies sentiments into Positive, Neutral, and Negative categories.
- Displays confidence scores for each predicted sentiment.
- Provides an interactive Flask web application for real-time analysis.
- Exposes a REST API for sentiment prediction.

---

## Project Architecture

```
Student Course Review
          │
          ▼
Data Preprocessing
          │
          ▼
Aspect Extraction (spaCy)
          │
          ▼
Transformer-Based Sentiment Analysis
(DeBERTa + RoBERTa)
          │
          ▼
Sentiment Prediction
          │
          ▼
Flask Backend
          │
          ▼
Interactive Web Interface
```

---

## Tech Stack

### Programming Language

- Python

### Machine Learning & NLP

- spaCy
- Hugging Face Transformers
- DeBERTa
- RoBERTa
- PyTorch

### Backend

- Flask
- Flask-CORS

### Data Processing

- Pandas
- NumPy

---

## Dataset

The project uses a dataset of student course reviews to perform aspect-level sentiment analysis.

Example aspects extracted include:

- Instructor
- Assignments
- Examinations
- Syllabus
- Course Content

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

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

---

## Note

Pretrained transformer model files are **not included** in this repository because they exceed GitHub's file size limit. Before running the application, download or train the required DeBERTa and RoBERTa models and place them in the appropriate project directory.

---

## Usage

Run the Flask server:

```bash
python server.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

Enter a course review, and the system will extract relevant aspects and predict the sentiment for each aspect.

---

## Project Structure

```
course-review-absa/
│
├── static/
├── templates/
├── aspect_extractor.py
├── sentiment_model.py
├── server.py
├── requirements.txt
├── README.md
└── ...
```

---

## Results

The application is capable of:

- Extracting multiple aspects from a single review.
- Predicting aspect-level sentiment using transformer models.
- Displaying confidence scores for each detected aspect.
- Providing real-time sentiment analysis through a web interface.

---

## Future Improvements

- Support multilingual course reviews.
- Improve aspect extraction using dependency parsing.
- Deploy the application using Docker and cloud platforms.
- Add batch review analysis through file upload.
- Integrate advanced analytics and reporting dashboards.

---

## Author

**Hima Paul**

Final Year B.Tech Computer Science Engineering Student

Interested in Artificial Intelligence, Machine Learning, Computer Vision, and Natural Language Processing.
