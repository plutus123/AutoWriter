from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_name_or_path = "./gpt2-finetuned-bookcorpus"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

# Move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    num_sequences = int(request.form.get('num_sequences', 3))
    max_length = int(request.form.get('max_length', 50))

    # Prepare the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate completions with adjusted parameters
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=10,
        penalty_alpha=0.6,  # Controls the degeneration penalty (for contrastive search)
        top_k=4,  # Limits to top_k tokens
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the outputs
    completions = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        completions.append(text)

    return jsonify({'completions': completions})

# Placeholder route for collecting human feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    prompt = data['prompt']
    selected_completion = data['selected_completion']
    edited_completion = data.get('edited_completion', None)

    # TODO: Save the feedback data to a database or file for future fine-tuning
    # For now, we'll just print it to the console
    print("Prompt:", prompt)
    print("Selected Completion:", selected_completion)
    if edited_completion:
        print("Edited Completion:", edited_completion)

    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
