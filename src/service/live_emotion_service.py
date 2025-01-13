import os

current_emotion = "neutral"  # TODO: persist this


def get_all():
    return {
        'initial_emotion': current_emotion,
        'initial_image_src': os.path.join('static', f"{current_emotion}.svg")
    }
