import gradio as gr
import os
import torch

from model import SigCNN, signature_transform
from timeit import default_timer as timer

# Setup class names
class_names = ['forgery', 'original']

# Create SigCNN model
cnn_model = SigCNN(1, 2)

# Load saved model
cnn_model.load_state_dict(torch.load(f="CEDAR_model.pt", map_location=torch.device("cpu")))

# Create predict function
def predict(img):
    """Transforms and performs a prediction on img and returns prediction and time taken."""
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = signature_transform(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    cnn_model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(cnn_model(img), dim=1)
    
    # Create a dictionary with prediction label and prediction probability for each prediction class (Reqd. for Gradio)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and time 
    return pred_labels_and_probs, pred_time

# Create title, description and article strings
title = "Signature Verification System"
description = "The Signature Verification Project, made with PyTorch and trained on the CEDAR dataset, aims to develop a deep learning model capable of differentiating between genuine and forged signatures, enhancing security and trust in signature verification systems. Project strives to achieve high accuracy and reliability in real-world signature authentication scenarios."


article = "Created by Dasari Sai Sri Divya"

# Create examples list
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio app
sig_app = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=2, label="Predictions"),
                             gr.Number(label="Prediction Time (s)")],
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch
sig_app.launch()