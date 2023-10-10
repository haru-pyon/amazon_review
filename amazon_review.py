#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# Data: "Amazon Fine Food Reviews" at Kaggle
# https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
# License: Public Domain
df = pd.read_csv('Reviews.csv')


# In[ ]:


# import tqdm to visualize progressing
from tqdm import tqdm, tqdm_notebook


# # Clean Data

# In[ ]:


# Drop columns which will not be used
df.drop(["UserId", "ProfileName", "ProductId", "Time"], axis=1,inplace=True)


# In[ ]:


# Drop duplicated rows
df = df.drop_duplicates(keep='first', subset=["Score", "Summary","Text"])


# In[ ]:


# lower the texts
df["Summary"] = df["Summary"].str.lower()
df["Text"] = df["Text"].str.lower()


# In[ ]:


# Remove rows containing "dog" or "cat" in either the "Summary" or "Text" column
df = df[~(df['Summary'].str.contains('dog|cat|puppy', case=False) | df['Text'].str.contains('dog|cat|puppy', case=False))]


# In[ ]:


# Remove rows containing "shampoo" and so on in either the "Summary" or "Text" column
df = df[~(df['Summary'].str.contains('shampoo|soap|body wash|skin care|medicine|medication', case=False) | df['Text'].str.contains('shampoo|soap|body wash|skin care|medicine|medication', case=False))]


# In[ ]:


import re
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    return text

# Apply the clean_text function to the 'Text' column
tqdm.pandas(desc="Processing")
df['Text'] = df['Text'].progress_apply(clean_text)


# # Create Food Category Column based on the Text and Summary

# In[ ]:


# import vocabulary library
import nltk

# create food list from the library
from nltk.corpus import wordnet as wn
food = wn.synset('food.n.02')
food_lst = list(set([w for s in food.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))


# In[ ]:


# Replace underscores to spaces in the food_lst
food_list = [food.replace("_", " ") for food in food_lst]


# In[ ]:


# lower the food in the list
food_lower_lst = [food.lower() for food in food_list]


# In[ ]:


# function to extract food names from text using the food list 

def extract_food_category(text):
    food_category = []
    for food in food_lower_lst:
        if food in text:
            food_category.append(food)
    return ", ".join(food_category) if food_category else None


# In[ ]:


tqdm.pandas(desc="Processing")
df["Food_Category"] = df["Text"].progress_apply(extract_food_category)


# In[ ]:


# Define a function to remove non-string elements from text
def remove_non_string(text):
    # Use regular expressions to remove non-string elements
    return re.sub(r'[^a-zA-Z\s]', '', str(text))
df["Summary"] = df["Summary"].apply(remove_non_string)


# In[ ]:


tqdm.pandas(desc="Processing")
df["Food_Category_sum"] = df["Summary"].progress_apply(extract_food_category)


# In[ ]:


# Fill missing values with empty strings in both columns 
df["Food_Category"].fillna("", inplace=True)
df["Food_Category_sum"].fillna("", inplace=True)

# Combine text from both columns into a new column
df["Combined"] = df["Food_Category"] + ", " + df["Food_Category_sum"]


# In[ ]:


# Function to remove duplicate food names separated by commas
def remove_duplicates(text):
    words = text.split(', ')  # Split the text into words by comma and space
    unique_words = list(set(words))  # Convert to a set to remove duplicates and then back to a list
    cleaned_text = ', '.join(unique_words)  # Join the unique words back with comma and space
    return cleaned_text


# In[ ]:


# Apply the remove_duplicates function to the "Combined" column
tqdm.pandas(desc="Processing")
df["Combined"] = df["Combined"].apply(remove_duplicates)

df.drop(["Food_Category", "Food_Category_sum"], axis=1, inplace =True)
# Rename the 'Combined' to 'Food_Cat'
df.rename(columns={"Combined": "Food_Cat"}, inplace=True)


# In[ ]:


# Function to remove leading ", " from text
def remove_leading_comma_space(text):
    return text.lstrip(', ')

# Apply the remove_leading_comma_space function to the Food_Cat column
df['Food_Cat'] = df['Food_Cat'].apply(remove_leading_comma_space)


# # Create Word Cloud

# In[3]:


#  Remove stop words from Text


# In[ ]:


import spacy
nlp = spacy.load("en_core_web_sm")
def remove_stopwords(text):
    doc = nlp(text)
    non_stop_words = [token.text for token in doc if not token.is_stop]
    return " ".join(non_stop_words)

# Apply the remove_stopwords function
tqdm.pandas(desc="Processing")
df["Text"] = df["Text"].progress_apply(remove_stopwords)


# # Bag of Words (BoW) approach, specifically a CountVectorizer.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer 


# In[ ]:


# Separate the text based on scores (4 and 5 as positive, 1 and 2 as negative)
positive_text = ' '.join(df[df['Score'] > 3]['Text'])
negative_text = ' '.join(df[df['Score'] < 3]['Text'])


# # Unigrams

# In[ ]:


# Create CountVectorizer instances for positive and negative text
vectorizer_positive = CountVectorizer()
vectorizer_negative = CountVectorizer()

# Fit and transform the text data with CountVectorizer
X_positive = vectorizer_positive.fit_transform([positive_text])
X_negative = vectorizer_negative.fit_transform([negative_text])

# Get the feature names (words)
feature_names_positive = vectorizer_positive.get_feature_names_out()
feature_names_negative = vectorizer_negative.get_feature_names_out()


# In[ ]:


# Generate word clouds for positive and negative text
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(feature_names_positive, X_positive.toarray()[0])))
negative_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(dict(zip(feature_names_negative, X_negative.toarray()[0])))


# In[ ]:


# Display the word clouds
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Scores (Unigrams)', fontsize = 20)
plt.axis('off')
plt.show()


# In[ ]:


# Display the word clouds
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Negative Scores (Unigrams)', fontsize = 20)
plt.axis('off')
plt.show()


# # Bigrams

# In[ ]:


# Create CountVectorizer instances for positive and negative text with bigrams
vectorizer_positive = CountVectorizer(ngram_range=(2, 2))  # Use bigrams
vectorizer_negative = CountVectorizer(ngram_range=(2, 2))  # Use bigrams

# Fit and transform the text data with CountVectorizer for bigrams
X_positive = vectorizer_positive.fit_transform([positive_text])
X_negative = vectorizer_negative.fit_transform([negative_text])

# Get the feature names (bigrams)
feature_names_positive = vectorizer_positive.get_feature_names_out()
feature_names_negative = vectorizer_negative.get_feature_names_out()

# Generate word clouds for positive and negative text using bigrams
positive_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(dict(zip(feature_names_positive, X_positive.toarray()[0])))
negative_wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(dict(zip(feature_names_negative, X_negative.toarray()[0])))


# In[ ]:


# Display the word clouds
positive_wordcloud = WordCloud(width=800, height=400, collocations=False, colormap='viridis', background_color='white').generate_from_frequencies(dict(zip(feature_names_positive, X_positive.toarray()[0])))
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Scores (Bigrams)', fontsize = 20)
plt.axis('off')
plt.show()


# In[ ]:


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Negative Scores (Bigrams)', fontsize = 20)
plt.axis('off')
plt.show()


# # Create Bar plot for Trigrams

# In[ ]:


# Remove Punctuation
import string
def remove_punctuation(text):
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

df["Text"] = df["Text"].apply(remove_punctuation)


# In[ ]:


# Calculate trigram frequencies for positive and negative texts
positive_trigram_freq = Counter(zip(positive_text.split(), positive_text.split()[1:], positive_text.split()[2:]))
negative_trigram_freq = Counter(zip(negative_text.split(), negative_text.split()[1:], negative_text.split()[2:]))

# Calculate score differences for each trigram
score_differences = {}
for trigram in positive_trigram_freq.keys():
    if trigram in negative_trigram_freq:
        score_differences[trigram] = positive_trigram_freq[trigram] - negative_trigram_freq[trigram]

# Sort trigrams by score differences (from most positive to most negative)
sorted_trigrams = sorted(score_differences.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


# Extract the top N trigrams and their score differences
top_trigrams = [trigram[0] for trigram in sorted_trigrams[:30]]  # Show top 30

# Create a DataFrame for the top trigrams and their score differences
top_trigrams_df = pd.DataFrame({'Trigram': top_trigrams, 'Score Difference': [score_differences[trigram] for trigram in top_trigrams]})

# Create a Seaborn bar plot
plt.figure(figsize=(18, 10))
sns.barplot(edgecolor = "black", x='Score Difference', y='Trigram', data=top_trigrams_df, palette='viridis')
plt.title('Trigrams Affecting Higher Scores More than Lower Scores', fontsize=25)
plt.xlabel('Score Difference', fontsize =20)
plt.ylabel('Trigram',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()


# In[ ]:


# Extract the top N trigrams and their score differences for Negative Words
bottom_trigrams = [trigram[0] for trigram in sorted_trigrams[-30:]]  # Top 30 Negative

# Create a DataFrame for the top trigrams and their score differences
bottom_trigrams_df = pd.DataFrame({'Trigram': bottom_trigrams, 'Score Difference': [score_differences[trigram] for trigram in bottom_trigrams]})

# Create a Seaborn bar plot
plt.figure(figsize=(18, 10))
sns.barplot(edgecolor = "black", x='Score Difference', y='Trigram', data=bottom_trigrams_df, palette='inferno')
plt.title('Trigrams Affecting Lower Scores More than Higher Scores', fontsize=25)
plt.xlabel('Score Difference', fontsize =20)
plt.ylabel('Trigram',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()


# In[ ]:


# Create Word Cloud for Trigrams
# Define words to exclude from the word cloud
words_to_exclude = []  # Replace with the words you want to exclude

# Remove the specific words from the text
positive_text = ' '.join([word for word in positive_text.split() if word not in words_to_exclude])

# Create CountVectorizer instance for text with bigrams
vectorizer_positive = CountVectorizer(ngram_range=(3, 3))  # Use Trigrams

# Fit and transform the preprocessed text data with CountVectorizer for Trigrams
X_positive = vectorizer_positive.fit_transform([positive_text])

# Get the feature names
feature_names_positive = vectorizer_positive.get_feature_names_out()

# Generate the Positive word cloud without the excluded words
positive_wordcloud = WordCloud(width=800, height=400, collocations=False, background_color='white').generate_from_frequencies(dict(zip(feature_names_positive, X_positive.toarray()[0])))


# In[ ]:


plt.figure(figsize=(30, 15))
plt.subplot(1, 2, 2)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Scores (Trigram)', fontsize = 25)
plt.axis('off')
plt.show()


# In[ ]:


# Define words to exclude from the negative word cloud
words_to_exclude = []  # Replace with the words you want to exclude

# Remove the specific words from the negative text
negative_text = ' '.join([word for word in negative_text.split() if word not in words_to_exclude])

# Create CountVectorizer instance for negative text with bigrams
vectorizer_negative = CountVectorizer(ngram_range=(3, 3))  # Use bigrams

# Fit and transform the preprocessed negative text data with CountVectorizer for bigrams
X_negative = vectorizer_negative.fit_transform([negative_text])

# Get the feature names (bigrams)
feature_names_negative = vectorizer_negative.get_feature_names_out()
# Generate the negative word cloud without the excluded words
negative_wordcloud = WordCloud(width=800, height=400, collocations=False).generate_from_frequencies(dict(zip(feature_names_negative, X_negative.toarray()[0])))


# In[ ]:


plt.figure(figsize=(30, 15))
plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Negative Scores (Trigram)', fontsize = 25)
plt.axis('off')
plt.show()


# # Create Machine Learning Model

# In[ ]:


# Create a Data Frame which all Scores (1-4) are well balanced distributed
score_counts = df['Score'].value_counts()

# Find the minimum count among the scores
min_count = score_counts.min()

# Create an empty DataFrame to store balanced data
balanced_df = pd.DataFrame(columns=df.columns)

# Iterate through each score and sample 'min_count' samples
for score in score_counts.index:
    score_data = df[df['Score'] == score]
    balanced_data = score_data.sample(min_count, random_state=42)  # Use a fixed seed for reproducibility
    balanced_df = pd.concat([balanced_df, balanced_data])

# Shuffle the rows to randomize the order
balanced_df = balanced_df.sample(frac=1, random_state=42)


# In[ ]:


# Drop Columns which will not be used
df.drop("HelpfulnessNumerator", axis=1,inplace=True)
df.drop("HelpfulnessDenominator", axis=1,inplace=True)


# In[ ]:


import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


# In[ ]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Score'], test_size=0.2, random_state=42)


# In[ ]:


# Tokenize the Text
# Initialize the Albert tokenizer
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# Tokenize and encode data
def tokenize_data(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in tqdm(texts, desc="Tokenizing"):
        encoding = tokenizer.encode_plus(
            text,
            max_length=max_length,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids.append(encoding["input_ids"])
        attention_masks.append(encoding["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


# In[ ]:


max_length = 100  # Limit the numbers of words to be input to the model

input_ids_train, attention_masks_train = tokenize_data(X_train, tokenizer, max_length)
input_ids_test, attention_masks_test = tokenize_data(X_test, tokenizer, max_length)


# In[ ]:


# Since the Score starts from 1 and not from 0, map and subtract 1 from the labels to match with Albert Model Labeling
labels_train = torch.tensor([label - 1 for label in y_train], dtype=torch.long)
labels_test = torch.tensor([label - 1 for label in y_test], dtype=torch.long)


# In[ ]:


# Create DataLoaders for training and testing
batch_size = 8
train_data = TensorDataset(input_ids_train, attention_masks_train, labels_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = TensorDataset(input_ids_test, attention_masks_test, labels_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# In[ ]:


# Load the pre-trained AlbertForSequenceClassification model
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", 
                                                        num_labels=5)
# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# If multiple GPUs are available, use DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Move the model to the selected device
model.to(device)


# In[ ]:


# Hyperparameter Tune
# Define a range of hyperparameters to tune
learning_rates = [2e-5, 3e-5, 5e-5]
num_epochs_values = [1, 2, 3]

# Initialize variables to store the best hyperparameters and the corresponding best validation score
best_hyperparameters = {'lr': None, 'num_epochs': None}
best_validation_score = float('inf')  # Initialize with a large value

# Training loop to perform hyperparameter tuning with tqdm
for lr in tqdm(learning_rates, desc="Tuning Learning Rate"):
    for num_epochs in tqdm(num_epochs_values, desc="Tuning Num Epochs"):
        
        # Define optimizer and learning rate scheduler
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(input_ids_train) * num_epochs)

        # Training loop
        for epoch in tqdm(range(num_epochs), desc=f"Training LR={lr}, Epochs={num_epochs}"):
            model.train()
            # Add your training code here

         # calculate validation loss
        validation_score = 0.0 

        # Check if the current hyperparameters result in a better validation score
        if validation_score < best_validation_score:
            best_validation_score = validation_score
            best_hyperparameters['lr'] = lr
            best_hyperparameters['num_epochs'] = num_epochs

# Print the best hyperparameters
print("Best Hyperparameters:", best_hyperparameters)


# In[ ]:


# Train Albert BERT Model with Training Data
import torch.nn as nn

# Define Training parameters
num_epochs = 1
lr = 2e-5
accumulation_steps = 2  # Gradient accumulation steps

# Define the optimizer and learning rate scheduler
model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=5)
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * num_epochs
)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# If multiple GPUs are available, use DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Move the model to the selected device
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    true_labels = []
    predicted_labels_list = []
    accumulation_steps_loss = 0.0  # Accumulated loss for gradient accumulation

    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")):
        input_ids, attention_masks, labels = batch
        input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits

        loss = criterion(logits, labels) / accumulation_steps  # Divide the loss by accumulation steps
        total_loss += loss.item()
        accumulation_steps_loss += loss.item()

        loss.backward()
        
        # Perform gradient accumulation every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accumulation_steps_loss = 0.0  # Reset accumulated loss

        predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
        true_labels.extend(labels.cpu().numpy())
        predicted_labels_list.extend(predicted_labels)

    accuracy = accuracy_score(true_labels, predicted_labels_list)
    report = classification_report(
        true_labels,
        predicted_labels_list,
        labels=[0, 1, 2, 3, 4],
        target_names=["Score 1", "Score 2", "Score 3", "Score 4", "Score 5"]
    )

    print(f"Epoch {epoch + 1} - Average Training Loss: {total_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

# Save the trained model
model.save_pretrained('albert_amazon_model')


# In[ ]:


# Check the baseline Accuracy for Training Data to compare with the above result
# Min score and Max score for "Score" column
min_score = 1
max_score = 5

# Number of samples in the train data
num_samples = len(y_train)

# Generate random guesses for the "Score" column
random_scores = np.random.randint(min_score, max_score + 1, size=num_samples)

# Calculate the accuracy of the random baseline
correct_predictions = (y_train == random_scores).sum()
accuracy = correct_predictions / num_samples

# Display the baseline accuracy
print(f"Random Baseline Accuracy: {accuracy:.4f}")


# In[ ]:


# Evaluate the Model with Test Data
# Testing loop
model.eval()  # Set the model to evaluation mode
test_true_labels = []
test_predicted_labels_list = []

for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
    input_ids, attention_masks, labels = batch
    input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)

    with torch.no_grad():  # Disable gradient calculation during testing
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits

    predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
    test_true_labels.extend(labels.cpu().numpy())
    test_predicted_labels_list.extend(predicted_labels)

test_accuracy = accuracy_score(test_true_labels, test_predicted_labels_list)
test_report = classification_report(
    test_true_labels,
    test_predicted_labels_list,
    labels=[0, 1, 2, 3, 4],
    target_names=["Score 1", "Score 2", "Score 3", "Score 4", "Score 5"]
)

print("Test Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(test_report)


# In[ ]:


# Check the baseline Accuracy for the test data
min_score = 1
max_score = 5

# Number of samples in the Test data
num_samples = len(y_test)

# Generate random guesses for the "Score" column
random_scores = np.random.randint(min_score, max_score + 1, size=num_samples)

# Calculate the accuracy of the random baseline
correct_predictions = (y_test == random_scores).sum()
accuracy = correct_predictions / num_samples

# Display the baseline accuracy
print(f"Random Baseline Accuracy: {accuracy:.4f}")


# # Use the Model for predicting the Score based on the user input text

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

# Print the top N words and their importance scores
top_n = 10  # Change to the number of top words you want to see
for word, score in word_scores[:top_n]:
    print(f"Word: {word}, Importance Score: {score:.4f}")

print("")
print(f"Predicted Score: {predicted_score}")


# In[ ]:




