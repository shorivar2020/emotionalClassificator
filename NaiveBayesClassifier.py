import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


anger_anger, anger_disgust, anger_fear, anger_joy, anger_sadness, anger_surprise = 0
disgust_anger, disgust_disgust, disgust_fear, disgust_joy, disgust_sadness, disgust_surprise = 0
fear_anger, fear_disgust, fear_fear, fear_joy, fear_sadness, fear_surprise = 0
joy_anger, joy_disgust, joy_fear, joy_joy, joy_sadness, joy_surprise = 0
sadness_anger, sadness_disgust, sadness_fear, sadness_joy, sadness_sadness, sadness_surprise = 0
surprise_anger, surprise_disgust, surprise_fear, surprise_joy, surprise_sadness, surprise_surprise = 0
def get_primary_emotion(emotion, emotion_map):
    for primary_emotion, similar_emotions in emotion_map.items():
        if emotion in similar_emotions:
            return primary_emotion
    return None


emotion_map = {
    "anger": ["anger", "annoyance", "disapproval", "hate"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "happy", "happiness", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "relief", "pride", "admiration", "desire", "caring", "relief", "enthusiasm", "fun"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse", "worry"],
    "surprise": ["surprise", "realization", "confusion", "curiosity", "anticipation"],
    "neutral": ["neutral"]
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

circumplex_model_valence = {
    "positive": ["joy", "happy", "amusement", "approval", "excitement", "gratitude", "love",
                 "optimism", "relief", "pride", "admiration", "desire", "caring", "curiosity", "happiness"],
    "neutral": ["neutral", "realization", "surprise", "anticipation", "neutral"],
    "negative": ["worry", "sadness", "disappointment", "embarrassment", "grief", "remorse", "anger",
                 "annoyance", "disapproval", "hate", "disgust", "fear", "confusion", "nervousness"]
}


def load_data(file_path):
    contents = []
    sentiments = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["Emotion"] == "":
                continue
            contents.append(row["Text"])
            sentiments.append(row["Emotion"])
    return contents, sentiments


def train_model(tweets, labels):
    print(tweets)
    print(labels)
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(tweets, labels)
    return model


def predict_emotion(model, text):
    return model.predict([text])




def compare_emotions_with_d3(model, file_path):
    match_count, mismatch_count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            tweet_text = row['Text']
            actual_emotion = row['Emotion']

            if actual_emotion == "empty":
                continue

            print(f"Have: {actual_emotion}")
            actual_emotion = get_primary_emotion(actual_emotion, emotion_map)
            predicted_emotion = predict_emotion(model, tweet_text)[0]
            predicted_emotion = get_primary_emotion(predicted_emotion, emotion_map)

            print(f"Predicted emotion: {predicted_emotion} Actual emotion: {actual_emotion}")
            predicted_emotion = get_primary_emotion(predicted_emotion, emotion_map)

            arousal = get_primary_emotion(predicted_emotion, circumplex_model_arousal)
            valence = get_primary_emotion(predicted_emotion, circumplex_model_valence)
            actual_emotion_arousal = get_primary_emotion(actual_emotion, circumplex_model_arousal)
            actual_emotion_valence = get_primary_emotion(actual_emotion, circumplex_model_valence)

            if (arousal == actual_emotion_arousal) and (valence == actual_emotion_valence):
                match_count += 1
                # print(f"Comparison: {'Match'}")
            else:
                mismatch_count += 1
            if actual_emotion == "anger":
                if predicted_emotion == "anger":
                    anger_anger = anger_anger + 1
                if predicted_emotion == "disgust":
                    anger_disgust = anger_disgust + 1
                if predicted_emotion == "fear":
                    anger_fear = anger_fear + 1
                if predicted_emotion == "joy":
                    anger_joy = anger_joy + 1
                if predicted_emotion == "sadness":
                    anger_sadness = anger_sadness + 1
                if predicted_emotion == "surprise":
                    anger_surprise = anger_surprise + 1
            if actual_emotion == "disgust":
                if predicted_emotion == "anger":
                    disgust_anger = disgust_anger + 1
                if predicted_emotion == "disgust":
                    disgust_disgust = disgust_disgust + 1
                if predicted_emotion == "fear":
                    disgust_fear = disgust_fear + 1
                if predicted_emotion == "joy":
                    disgust_joy = disgust_joy + 1
                if predicted_emotion == "sadness":
                    disgust_sadness = disgust_sadness + 1
                if predicted_emotion == "surprise":
                    disgust_surprise = disgust_surprise + 1
            if actual_emotion == "fear":
                if predicted_emotion == "anger":
                    fear_anger = fear_anger + 1
                if predicted_emotion == "disgust":
                    fear_disgust = fear_disgust + 1
                if predicted_emotion == "fear":
                    fear_fear = fear_fear + 1
                if predicted_emotion == "joy":
                    fear_joy = fear_joy + 1
                if predicted_emotion == "sadness":
                    fear_sadness = fear_sadness + 1
                if predicted_emotion == "surprise":
                    fear_surprise = fear_surprise + 1
            if actual_emotion == "joy":
                if predicted_emotion == "anger":
                    joy_anger = joy_anger + 1
                if predicted_emotion == "disgust":
                    joy_disgust = joy_disgust + 1
                if predicted_emotion == "fear":
                    joy_fear = joy_fear + 1
                if predicted_emotion == "joy":
                    joy_joy = joy_joy + 1
                if predicted_emotion == "sadness":
                    joy_sadness = joy_sadness + 1
                if predicted_emotion == "surprise":
                    joy_surprise = joy_surprise + 1
            if actual_emotion == "sadness":
                if predicted_emotion == "anger":
                    sadness_anger = sadness_anger + 1
                if predicted_emotion == "disgust":
                    sadness_disgust = sadness_disgust + 1
                if predicted_emotion == "fear":
                    sadness_fear = sadness_fear + 1
                if predicted_emotion == "joy":
                    sadness_joy = sadness_joy + 1
                if predicted_emotion == "sadness":
                    sadness_sadness = sadness_sadness + 1
                if predicted_emotion == "surprise":
                    sadness_surprise = sadness_surprise + 1
            if actual_emotion == "surprise":
                if predicted_emotion == "anger":
                    surprise_anger = surprise_anger + 1
                if predicted_emotion == "disgust":
                    surprise_disgust = surprise_disgust + 1
                if predicted_emotion == "fear":
                    surprise_fear = surprise_fear + 1
                if predicted_emotion == "joy":
                    surprise_joy = surprise_joy + 1
                if predicted_emotion == "sadness":
                    surprise_sadness = surprise_sadness + 1
                if predicted_emotion == "surprise":
                    surprise_surprise = surprise_surprise + 1
                print(f"Comparison: {'Mismatch'}")
                print(f"Tweet: {tweet_text}")
                print(f"Active-Passive: {arousal} Nego-Pos: {valence}")
                print(f"Active-Passive: {actual_emotion_arousal} Nego-Pos: {actual_emotion_valence}")
                print(f"Predicted emotion: {predicted_emotion} Actual emotion: {actual_emotion}")
            print("---")
            print(match_count)
            print(mismatch_count)
            print(f"Success:{match_count / (match_count + mismatch_count)}%")
        print(f"anger: {anger_anger} , {anger_disgust}, {anger_fear}, {anger_joy}, {anger_sadness}, {anger_surprise}")
        print(
            f"disgust: {disgust_anger} , {disgust_disgust}, {disgust_fear}, {disgust_joy}, {disgust_sadness}, {disgust_surprise}")
        print(f"fear: {fear_anger} , {fear_disgust}, {fear_fear}, {fear_joy}, {fear_sadness}, {fear_surprise}")
        print(f"joy: {joy_anger} , {joy_disgust}, {joy_fear}, {joy_joy}, {joy_sadness}, {joy_surprise}")
        print(
            f"sadness: {sadness_anger} , {sadness_disgust}, {sadness_fear}, {sadness_joy}, {sadness_sadness}, {sadness_surprise}")
        print(
            f"surprise: {surprise_anger} , {surprise_disgust}, {surprise_fear}, {surprise_joy}, {surprise_sadness}, {surprise_surprise}")


if __name__ == "__main__":
    tweets, labels = load_data('datasets/tweet_emotions.csv')
    model = train_model(tweets, labels)

    # test_text = "Pats in philly at 2 am. I love it. Mmm cheesesteak"
    # predicted_emotion = predict_emotion(model, test_text)
    # print("Predicted emotion:", predicted_emotion)

    compare_emotions_with_d3(model, 'datasets/d2.csv')
