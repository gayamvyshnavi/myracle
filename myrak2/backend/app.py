import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import openai
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize Flask app
app = Flask(__name__)

# OpenAI GPT-4 API key (replace with your own)
openai.api_key = 'api key'
# Directory to save uploaded screenshots
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists and recreate it for each new upload
def recreate_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        for file in os.listdir(UPLOAD_FOLDER):
            os.remove(os.path.join(UPLOAD_FOLDER, file))
    else:
        os.makedirs(UPLOAD_FOLDER)

# Load BLIP Processor and Model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    # Open and process the image
    image = Image.open(image_path).convert("RGB")
    # Generate caption using BLIP
    inputs = blip_processor(images=image, return_tensors="pt")
    generated_ids = blip_model.generate(pixel_values=inputs["pixel_values"])
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption

def generate_testing_instructions(captions, context):
    # Create the input text for GPT-4
    input_text = f"Here are the feature descriptions based on the screenshots:\n"
    
    # Add captions to the input text
    for i, caption in enumerate(captions):
        input_text += f"Screenshot {i+1}: {caption}\n"
    
    # Add optional context if provided
    if context:
        input_text += f"\nContext: {context}\n"
    
    # Generate the test instructions prompt
    input_text += """
Generate detailed test instructions for these features including:
1. Description
2. Pre-conditions
3. Testing Steps
4. Expected Result
"""

    try:
        # Call the GPT-4 API using ChatCompletion
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4 model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=500  # Adjust this as necessary
        )
        
        # Get the generated instructions
        generated_text = response['choices'][0]['message']['content']
        return generated_text
    
    except Exception as e:
        print(f"Error with GPT-4 API: {e}")
        return "There was an error generating instructions with the GPT-4 API."

# Route for generating testing instructions
@app.route('/generate-instructions', methods=['POST'])
def generate_instructions():
    # Recreate the uploads folder
    recreate_upload_folder()

    # Get the optional context
    context = request.form.get('context', '')

    # Get the uploaded image files
    files = request.files.getlist('screenshots')

    # Save uploaded images to the uploads folder
    image_paths = []
    for file in files:
        if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            image_paths.append(file_path)

    # If no valid images are uploaded, return an error
    if not image_paths:
        return jsonify({'error': 'No valid image files uploaded.'}), 400

    # Generate captions for each image
    captions = [generate_caption(image_path) for image_path in image_paths]

    # Generate testing instructions using GPT-4
    instructions = generate_testing_instructions(captions, context)
    print(instructions)

    # Return the generated instructions as a JSON response
    return jsonify({'instructions': instructions})

if __name__ == '__main__':
    app.run(debug=True)
