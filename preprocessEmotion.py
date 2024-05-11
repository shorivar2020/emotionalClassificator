def get_primary_emotion(emotion, emotion_map):
    for primary_emotion, similar_emotions in emotion_map.items():
        if emotion in similar_emotions:
            return primary_emotion
    return None

emotion_map = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"]
}

def main():
    emotion = "love"
    primary_emotion = get_primary_emotion(emotion.lower(), emotion_map)
    if primary_emotion:
        print(f"Main emotion: {primary_emotion}")

if __name__ == "__main__":
    main()
