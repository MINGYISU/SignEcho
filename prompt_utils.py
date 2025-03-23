import base64
import json

# Base prompt (as model configuration)
BASE_PROMPT = """You are a professional sign language recognition expert. Please carefully analyze the hand gesture features in each image and return the corresponding English word.
Return only the word itself, without any explanation. If unrecognizable, return "UNKNOWN".
Possible words include: MY, YOUR, NAME, WHAT, HELLO, GOODBYE, THANK, YOU, PLEASE, SORRY, YES, NO, HELP, EAT, DRINK, SLEEP, WORK, STUDY, PLAY, STOP"""

# Detailed instructions (as supplementary for each call)
DETAILED_PROMPT = """These images are continuous sign language actions, with each image representing a gesture.
Please follow these rules for recognition:
1. Analyze the main features of the gesture (such as hand position, shape, movement direction, etc.)
2. Consider the context and continuity of the gesture
3. Return "UNKNOWN" if the gesture is incomplete or unclear
4. Ensure recognition results match the standard sign language vocabulary"""

def encode_image_to_base64(image_path):
    """Convert an image file to base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gen_prompt(leng: int | None) -> str:
    with open('videos.jsonl', 'r') as f:
        dataset = json.load(f)
    if leng is None:
        leng = len(dataset)
    else:
        assert 0 < leng <= len(dataset)
    
    prompt_data = ""
    import random
    dataset = random.sample(dataset, leng)
    for item in dataset:
        base64_img = [encode_image_to_base64(img_path) for img_path in item['frames']]
        instruction = item['label']
        prompt_data.append({'image': base64_img, 'meaning': instruction})

    # generate content parts
    content_parts = [{"text": BASE_PROMPT, "text": DETAILED_PROMPT}]
    
    # Add each example with its image and meaning
    for example in prompt_data["examples"]:
        # Add the image
        content_parts.append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": example["image"]
            }
        })
        
        # Add the meaning
        content_parts.append({
            "text": f"This sign means: {example['meaning']}"
        })
    
    return content_parts