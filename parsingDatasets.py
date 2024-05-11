import pandas as pd

df = pd.read_csv("datasets/original/go_emotions_dataset.csv")

emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
            'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
            'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

# Datasets files
__dataset__ = "datasets/original/test_original.csv"

# Datasets tests
__datasetTest__ = "datasets/original/test_2.csv"

emotion = {
    "sadness": "0",
    "joy": "1",
    "love": "2",
    "anger": "3",
    "fear": "4"
}


def replace_emotion(text):
    for emo, code in emotion.items():
        if code == text:
            return emo
    return text


def parse_emotion_dataset():
    with open(__dataset__, 'r') as infile, open(__datasetTest__, 'w') as outfile:
        next(infile)  # skip header
        for line in infile:
            text, code = line.strip().split(',')
            replaced_code = replace_emotion(code)
            outfile.write(f"{text},{replaced_code}\n")


def find_emotion(row):
    for emotion in emotions:
        if row[emotion] == 1:
            return emotion
    return "No emotion"


if __name__ == "__main__":
    # Go emotions dataset
    df['emotion'] = df.apply(find_emotion, axis=1)
    df = df.drop(emotions, axis=1)
    df.to_csv('go_emotions.csv', index=False)
    # Emotion dataset for Emotion Recognition Tasks
    parse_emotion_dataset()
