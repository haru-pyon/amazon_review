#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Import necessary library


# In[ ]:


from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch


# In[ ]:


# Load the model and tokenizer
model = AlbertForSequenceClassification.from_pretrained("albert_amazon_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")


# In[ ]:


# Take user input
user_input = input("Enter text: ")
print("")

def predict_by_albert(user_input):
    # Tokenize the user input
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    
    # Get the model's predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted score
    predicted_score = torch.argmax(outputs.logits).item()
    # Add 1 as the Score should be 1 to 5 instead from 0 to 4
    predicted_score += 1
    
    # Retrieve the tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Calculate word importance scores based on the magnitude of word embeddings
    word_importance = torch.abs(model.get_input_embeddings()(inputs["input_ids"])).sum(dim=2).squeeze().tolist()
    
    # Pair words with their importance scores
    word_scores = list(zip(tokens, word_importance))
    
    # Sort words by importance (highest to lowest)
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return word_scores, predicted_score

def main_function(user_input):
    importance_list=""
    word_scores, predicted_score = predict_by_albert(user_input)
    # Print the top N words and their importance scores
    top_n = 5  # Change to the number of top words you want to see
    for word, score in word_scores[:top_n]:
        importance_list += f"Word: **{word}**, Importance Score: {score:.4f}\n\n"
    
    return importance_list, predicted_score

main_function(user_input)


# # Using Gradio 

# In[ ]:


import gradio as gr


# In[ ]:


with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.sky)) as demo:
    gr.Markdown(
        """
        # Data Analytics Bootcamp Final Project
        ## Supervised Machine Learning of Amazon Review 
        """
    )

    inp = gr.Textbox("Input Review Sentence here", label="Review")
    btn = gr.Button("Predict Review Score")

    importance = gr.Markdown(label="Important words analysis")
    score = gr.Textbox(label = "Predicted Score")
    btn.click(fn=main_function, inputs=inp, outputs=[importance, score])
demo.launch(share=True)

