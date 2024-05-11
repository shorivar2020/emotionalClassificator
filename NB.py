import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Datasets files
__dataset__ = "datasets/tweet_emotions.csv"
__dataset2__ = "datasets/emotion_dataset.csv"
__dataset3__ = "datasets/go_emotions.csv"
__dataset4__ = "datasets/d2.csv"
__dataset5__ = "datasets/d3.csv"
# Datasets tests
__datasetTest__ = "datasets/tweet_emotions_test.csv"
__dataset2Test__ = "datasets/emotion_dataset_test.csv"
__dataset3Test__ = "datasets/go_emotions_test.csv"
__dataset4Test__ = "datasets/d2_2.csv"
__dataset5Test__ = "datasets/d3_2.csv"
# Datasets header
__datasetText__ = 'Text'
__datasetEmotion__ = 'Emotion'

emotion_map = {
    "anger": ["anger", "annoyance", "disapproval", "hate"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "happy", "happiness", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "relief", "pride", "admiration", "desire", "caring", "relief", "enthusiasm", "fun"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse", "worry", "boredom"],
    "surprise": ["surprise", "realization", "confusion", "curiosity", "anticipation"],
    "neutral": ["neutral"],
    "empty": ["empty"]
}

circumplex_model_valence = {
    "positive": ["joy", "happy", "amusement", "approval", "excitement", "gratitude", "love",
                 "optimism", "relief", "pride", "admiration", "desire", "caring", "curiosity", "happiness"],
    "neutral": ["neutral", "realization", "surprise", "anticipation"],
    "negative": ["worry", "sadness", "disappointment", "embarrassment", "grief", "remorse", "anger",
                 "annoyance", "disapproval", "hate", "disgust", "fear", "confusion", "nervousness"]
}

circumplex_model_arousal = {
    "active": ["worry", "joy", "anger", "hate", "disgust", "fear", "surprise", "confusion", "interest", "embarrassment",
               "nervousness", "amusement", "pride", "excitement", "annoyance", "curiosity", "happy", "happiness"],
    "neutral": ["neutral"],
    "none": ["disappointment", "realization", "anticipation", "grief", "remorse",
             "disapproval", "approval", "gratitude", "love", "optimism",
             "relief", "admiration", "desire", "caring"],
    "passive": ["sadness", "calm", "relaxed", "tired", "boredom", "bored", "depressed", "calm", "comforted"]
}

def get_primary_emotion(emotion, emotion_map):
    for primary_emotion, similar_emotions in emotion_map.items():
        if emotion in similar_emotions:
            return primary_emotion
    return None

def load_data(file_paths):
    contents = []
    sentiments = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                sentiment = row.get(__datasetEmotion__)
                if sentiment in ["empty", "", "No emotion"]:
                    continue
                contents.append(row.get(__datasetText__))
                sentiments.append(get_primary_emotion(sentiment, emotion_map))
    return contents, sentiments

def train_model(col1, col2):
    return make_pipeline(CountVectorizer(), MultinomialNB()).fit(col1, col2)


def predict_emotion(model, text):
    return model.predict([text])

def count_emotion(predicted_emotion, dict):
    if predicted_emotion == "anger":
        dict['anger'] += 1
    if predicted_emotion == "disgust":
        dict['disgust'] += 1
    if predicted_emotion == "fear":
        dict['fear'] += 1
    if predicted_emotion == "joy":
        dict['joy'] += 1
    if predicted_emotion == "sadness":
        dict['sadness'] += 1
    if predicted_emotion == "surprise":
        dict['surprise'] += 1
    if predicted_emotion == "neutral":
        dict['neutral'] += 1
    return dict

def output_confuse_matrix(anger, disgust, fear, joy, sadness, surprise, neutral):
    print("Confuse matrix")
    print('         ang dis fea joy sad sur neu')
    print('anger   ', end=' ')
    for key, value in anger.items():
        print(value, end=' ')
    print('\ndisgust ', end=' ')
    for key, value in disgust.items():
        print(value, end=' ')
    print('\nfear    ', end=' ')
    for key, value in fear.items():
        print(value, end=' ')
    print('\njoy     ', end=' ')
    for key, value in joy.items():
        print(value, end=' ')
    print('\nsadness ', end=' ')
    for key, value in sadness.items():
        print(value, end=' ')
    print('\nsurprise', end=' ')
    for key, value in surprise.items():
        print(value, end=' ')
    print('\nneutral ', end=' ')
    for key, value in neutral.items():
        print(value, end=' ')

def write_confusion_matrix_to_excel(anger, disgust, fear, joy, sadness, surprise, neutral, success, excel_file):
    data = {
        'Emotion': ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral'],
        'anger': list(anger.values()),
        'disgust': list(disgust.values()),
        'fear': list(fear.values()),
        'joy': list(joy.values()),
        'sadness': list(sadness.values()),
        'surprise': list(surprise.values()),
        'neutral': list(neutral.values()),
        'Success Rate (%)': [success, None, None, None, None, None, None]

    }
    df = pd.DataFrame(data)
    df.to_excel(excel_file, index=False)
def test_predication(model, file_path, excel_file):
    match_count = 0
    mismatch_count = 0
    anger = {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'neutral': 0}
    disgust = {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'neutral': 0}
    fear = {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'neutral': 0}
    joy = {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'neutral': 0}
    sadness = {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'neutral': 0}
    surprise = {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'neutral': 0}
    neutral = {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0, 'surprise': 0, 'neutral': 0}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tweet_text = row['Text']
            actual_emotion = get_primary_emotion(row['Emotion'], emotion_map)
            if actual_emotion == "empty":
                continue
            predicted_emotion = predict_emotion(model, tweet_text)[0]
            predicted_emotion = get_primary_emotion(predicted_emotion, emotion_map)
            predicted_valence = get_primary_emotion(predicted_emotion, circumplex_model_valence)
            actual_valence = get_primary_emotion(actual_emotion, circumplex_model_valence)
            predicted_arousal = get_primary_emotion(predicted_emotion, circumplex_model_arousal)
            actual_arousal = get_primary_emotion(actual_emotion, circumplex_model_arousal)
            if predicted_valence == actual_valence and predicted_arousal == actual_arousal:
                match_count += 1
            else:
                mismatch_count += 1
            if actual_emotion == "anger":
                anger = count_emotion(predicted_emotion, anger)
            if actual_emotion == "disgust":
                disgust = count_emotion(predicted_emotion, disgust)
            if actual_emotion == "fear":
                fear = count_emotion(predicted_emotion, fear)
            if actual_emotion == "joy":
                joy = count_emotion(predicted_emotion, joy)
            if actual_emotion == "sadness":
                sadness = count_emotion(predicted_emotion, sadness)
            if actual_emotion == "surprise":
                surprise = count_emotion(predicted_emotion, surprise)
            if actual_emotion == "neutral":
                neutral = count_emotion(predicted_emotion, neutral)
                # print(f"Predicted emotion: {predicted_emotion} Actual emotion: {actual_emotion}")
        print("---")
        print(match_count)
        print(mismatch_count)
        success = match_count / (match_count + mismatch_count) * 100
        print(f"Success: {success}%")
        output_confuse_matrix(anger, disgust, fear, joy, sadness, surprise, neutral)
        write_confusion_matrix_to_excel(anger, disgust, fear, joy, sadness, surprise, neutral, success, excel_file)




if __name__ == "__main__":
    file_paths = [__dataset__, __dataset2__, __dataset3__, __dataset4__, __dataset5__]
    texts, emotions = load_data(file_paths)
    model = train_model(texts, emotions)
    test_predication(model, __datasetTest__,  'tweet_emotions_all.xlsx')
    test_predication(model, __dataset2Test__, 'emotion_dataset_all.xlsx')
    test_predication(model, __dataset3Test__, 'go_emotions_all.xlsx')
    test_predication(model, __dataset4Test__, 'd2_all.xlsx')
    test_predication(model, __dataset5Test__, 'd3_all.xlsx')

    file_paths = [__dataset__]
    texts, emotions = load_data(file_paths)
    model = train_model(texts, emotions)
    test_predication(model, __datasetTest__,  'tweet_emotions.xlsx')

    file_paths = [__dataset2__]
    texts, emotions = load_data(file_paths)
    model = train_model(texts, emotions)
    test_predication(model, __dataset2Test__,  'emotion_dataset.xlsx')

    file_paths = [__dataset3__]
    texts, emotions = load_data(file_paths)
    model = train_model(texts, emotions)
    test_predication(model, __dataset3Test__,  'go_emotions.xlsx')

    file_paths = [__dataset4__]
    texts, emotions = load_data(file_paths)
    model = train_model(texts, emotions)
    test_predication(model, __dataset4Test__,  'd2.xlsx')

    file_paths = [__dataset5__]
    texts, emotions = load_data(file_paths)
    model = train_model(texts, emotions)
    test_predication(model, __dataset5Test__,  'd3.xlsx')