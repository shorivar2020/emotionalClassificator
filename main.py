from nrclex import NRCLex
import csv


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
    "joy": ["joy", "positive", "happy", "happiness", "amusement", "approval", "excitement", "gratitude", "love",
            "optimism", "relief", "pride", "admiration", "desire", "caring", "relief", "enthusiasm", "approval", "fun"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse", "worry", "boredom"],
    "surprise": ["surprise", "realization", "confusion", "curiosity", "anticipation"],
    "neutral": ["neutral"],
    "empty": ["empty"]
}

def get_primary_emotion(emotion, emotion_map):
    for primary_emotion, similar_emotions in emotion_map.items():
        if emotion in similar_emotions:
            return primary_emotion
    return None

def process_csv(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data_dict[row[__datasetText__]] = row[__datasetEmotion__]
    return data_dict


def analyze_emotions(content):
        text_object = NRCLex(content)
        return text_object.top_emotions


def check_emotion(emotion_list, target_emotion):
    for emotion, percentage in emotion_list:
        if emotion == target_emotion:
            return percentage
    return 0


def check_valence(emotion_list):
    valence = ""
    count = 0
    np_count = 0
    neutral_count = 0
    for emotion, percentage in emotion_list:
        if valence == "" or valence == "None" or valence == "trust":
            valence = get_primary_emotion(emotion, circumplex_model_valence)
        else:
            new_valence = get_primary_emotion(emotion, circumplex_model_valence)
            if valence == new_valence:
                count += 1
            else:
                if (valence == "negative" and new_valence == "positive") or (valence == "positive" and new_valence == "negative"):
                    np_count += 1
                elif valence == "neutral" or new_valence == "neutral":
                    print(emotion_list, valence, new_valence)
                    neutral_count += 1
                else:
                    print(emotion_list, valence, new_valence)
            valence = new_valence
    if neutral_count > 0 :
        print(count, neutral_count)
    if count == len(emotion_list) - 1:
        return len(emotion_list) - 1 - count
    # else:
    #     print("Checked ", count, len(emotion_list))

circumplex_model_valence = {
    "positive": ["positive", "joy", "happy", "amusement", "approval", "excitement", "gratitude", "love",
                 "optimism", "relief", "pride", "admiration", "desire", "caring", "curiosity", "happiness"],
    "neutral": ["trust", "neutral", "realization", "surprise", "anticipation"],
    "negative": ["negative", "worry", "sadness", "disappointment", "embarrassment", "grief", "remorse", "anger",
                 "annoyance", "disapproval", "hate", "disgust", "fear", "confusion", "nervousness"]
}


def save_to_csv(data_dict, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['content', 'source_emotions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Best 0.5 - 1
        match_counter_high = 0
        # 0.2 - 0.5
        match_counter_mid = 0
        # 0.0 - 0.2
        match_counter_low = 0
        mismatch_counter = 0
        for key, content in data_dict.items():
            if data_dict[key] == "empty":
                continue
            if data_dict[key] == "":
                continue
            if data_dict[key] == "No emotion":
                continue
            check_valence(analyze_emotions(content))
            # check_arousal()
            percentage = check_emotion(analyze_emotions(content), get_primary_emotion(data_dict[key], emotion_map))
            if percentage > 0:
                # print(percentage)
                if percentage >= 0.5:
                    match_counter_high += 1
                    # print(analyze_emotions(content), data_dict[key])
                elif percentage > 0.2:
                    match_counter_mid += 1
                else:
                    match_counter_low += 1
                # print(analyze_emotions(content), data_dict[key])
            else:
                mismatch_counter += 1
            writer.writerow({
                'content': content,
                'source_emotions': data_dict[key]
            })

        all_match = match_counter_high+match_counter_mid+match_counter_low
        success = all_match / (all_match + mismatch_counter) * 100
        print(f"High 0.5 - 1: {match_counter_high} | {match_counter_high / (all_match + mismatch_counter) * 100}%")
        print(f"Mid 0.2 - 0.5: {match_counter_mid} | {match_counter_mid / (all_match + mismatch_counter) * 100}%")
        print(f"Low 0.0 - 0.2: {match_counter_low} | {match_counter_low / (all_match + mismatch_counter) * 100}%")
        print(f"Mis 0.0: {mismatch_counter} | {mismatch_counter / (all_match + mismatch_counter) * 100}%")
        print(f"Success: {success}%")


if __name__ == '__main__':
    result_dict = process_csv(__dataset__)
    save_to_csv(result_dict, "tweets_nrc.csv")
