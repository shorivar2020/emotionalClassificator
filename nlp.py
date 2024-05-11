import csv

def process_csv(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data_dict[row['content']] = row['sentiment']
    return data_dict

def process_csv_d2(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data_dict[row['Text']] = row['Emotion']
    return data_dict

def process_csv_d3(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            # Extract 'Text' and 'emotion' columns
            text = row['Text']
            emotion = row['emotion']
            # Store in dictionary
            data_dict[text] = emotion
    return data_dict

def print_contents(data_dict):
    for content in data_dict.keys():
        print(content)
        emotions = analyze_emotions(content, lexicon)
        print(emotions)
        print(data_dict.get(content))
        print(calculate_StSc(emotions))



def save_to_csv(data_dict, lexicon, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['content', 'my_analyze_emotions', 'source_emotions', 'StSc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for content in data_dict.keys():
            emotions = analyze_emotions(content, lexicon)
            writer.writerow({
                'content': content,
                'my_analyze_emotions': emotions,
                'source_emotions': data_dict.get(content),
                'StSc': calculate_StSc(emotions)
            })
def load_lexicon(lexicon_file):
    lexicon = {}
    with open(lexicon_file, 'r', encoding='utf-8') as file:
        for line in file:
            word, emotion, value = line.strip().split('\t')
            if word in lexicon:
                lexicon[word].append((emotion, int(value)))
            else:
                lexicon[word] = [(emotion, int(value))]
    return lexicon


# def analyze_emotions(text, lexicon):
#     emotions = {emotion: 0 for emotion in
#                 ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]}
#
#     for word in text.split():
#         word = word.lower()
#         if word in lexicon:
#             for emotion, value in lexicon[word]:
#                 if value == 1 and emotion in emotions:
#                     emotions[emotion] += 1
#
#     return emotions

def analyze_emotions(text, lexicon):
    emotions = {emotion: 0 for emotion in
                ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]}

    for word in text.split():
        word = word.lower()
        if word in lexicon:
            for emotion, value in lexicon[word]:
                if value == 1 and emotion in emotions:
                    emotions[emotion] += 1

    non_zero_emotions = {emotion: count for emotion, count in emotions.items() if count > 0}
    return non_zero_emotions
def calculate_StSc(emotions):
    # Initialize positive and negative counts to handle missing keys
    positive_count = emotions.get('joy', 0) + emotions.get('anticipation', 0) + emotions.get('trust', 0) + emotions.get('surprise', 0)
    negative_count = emotions.get('anger', 0) + emotions.get('disgust', 0) + emotions.get('fear', 0) + emotions.get('sadness', 0)

    total_words = sum(emotions.values())

    StSc = (positive_count - negative_count) / total_words if total_words > 0 else 0

    return StSc

output_file = 'd2_res.csv'
file_path = 'datasets/d2.csv'
lexicon_file = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
lexicon = load_lexicon(lexicon_file)
result_dict = process_csv_d3(file_path)
# print(result_dict)
# result_dict = process_csv(file_path)
# print_contents(result_dict)
save_to_csv(result_dict, lexicon, output_file)


# text = "love you, but i hate you"
# text2 = "i don know you, but i love you"
# emotions = analyze_emotions(text, lexicon)
# print(emotions)
# StSc = calculate_StSc(emotions)
# print("StSc:", StSc)