#!/usr/bin/env python
# coding: utf-8

# # Strategic Showcase: Leveraging Unsupervised Aspect Sentiment Analysis for Optimal Prime Day Product Selection
# ### Problem Statement  
# In today's online retail environment, events like Prime Day create significant business value. Optimizing product recommendations for these events can drive better user experience and increase sales. Through the lens of unsupervised aspect-based sentiment analysis, this project aims to curate products that resonate best with customer sentiments, particularly concerning their price value perceptions.   
# 
# **Keywords**: Sentiment Analysis, Aspect-Based Analysis, Unsupervised Learning, Prime Day Optimization, Strategic Recommendations. 
# ### Data Preprocessing  (rewrite)
# The dataset contains a total of 6939 image files divided equally into 3 classes (covid, pneumonia, normal)  
# Resized into 128 x 128 pixels in order to reduce computational complexity  
# Split the data into 80% train and 20% test sets   
# Create a model with transfer learning  
# 
# ### Methodology (rewrite)
# Transfer learning is: model developed for a task is reused as the starting point for a model on a second task  
# Introduce the Deep Neural Networks you used in your project  
# 
# Model 1: Plain VGG 19 model adapted adding an output layer  
# Model 2: VGG19 Finetuning adding additional conv2D layers  
# Model 3: VG19 Extended Plus Data Augmentation  
# **Keywords**: multi-label classification  
# 
# ### MultiClass Evaluation against Extended VGG19 Model  (re-write)
# Recall is the porpotion of the positive is corectly classified which is true positive over all positive (true positive + false negative)  
# Precision is the porpotion of predicted positives is truly positive which is true positive over the all predicted positive  
# F1 is the weighted average of precision and recall, which is a combination metrics  
# 
# ### Issues / Improvements  
# Might not capture complex sentiments or sarcasm.  
# Lack of Human-annotated data for aspect-based sentiments  
# Need support from A/B Testing for its business incremental value
# 
# ### References  
# https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data
# 
# https://github.com/sloria/TextBlob
# 
# https://www.amazon.com/Prime-Day/b?node=13887280011 (For understanding Prime Day dynamics).

# In[1]:


import pandas as pd
import sqlite3


# In[2]:


con = sqlite3.connect('C:\\project\\amazon_food\\database.sqlite')


# In[3]:


review_df = pd.read_sql_query(""" SELECT * FROM Reviews""", con)


# In[4]:


#print(review_df)


# In[5]:


#print(review_df.shape)


# # Data Cleaning

# In[6]:


# Drop duplicate reviews, same review can't be entered at same time point
df_final = review_df.drop_duplicates(subset={"UserId","ProfileName","Time","Text"})
print(df_final.shape)

# Helpfulness Numerator (Number of people finding this product useful)
# Helpfulness Denominator (Number of people finding this product useful + nimber of people not finding this profuct useful)
# Checking if Helpfulness Numeratior <= Helpfulness Denominator
df_final = df_final[(df_final['HelpfulnessNumerator']<=df_final['HelpfulnessDenominator'])]
print(df_final.shape)

df_final = df_final[df_final["Score"]!=3]
#print(df_final.shape)


# In[7]:


## take 1% sample to make sure it runs first
df_sampled = df_final.sample(frac=0.01, random_state=42)
#print(df_sampled.shape)


# ### Aspect Extraction

# In[8]:


from textblob import TextBlob

def extract_aspects(text):
    return [np for np in TextBlob(text).noun_phrases]

df_sampled['aspects'] = df_sampled['Text'].apply(extract_aspects)


# In[9]:


#df_sampled


# In[ ]:


import matplotlib.pyplot as plt

aspect_freq = {}
for _, row in df_sampled.iterrows():
    for aspect in row['aspects']:
        aspect_freq[aspect] = aspect_freq.get(aspect, 0) + 1


# In[ ]:


#aspect_freq


# In[ ]:


price_value_aspects = {k: v for k, v in aspect_freq.items() if 'price' in k or 'value' in k}


# In[ ]:


#price_value_aspects


# In[ ]:


#from wordcloud import WordCloud
#import matplotlib.pyplot as plt

#wordcloud = WordCloud(background_color="white", width=1000, height=500).generate_from_frequencies(price_value_aspects)
#plt.figure(figsize=(10, 5))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis('off')
#plt.show()


# ### Base Model

# In[ ]:


def aspect_sentiment_base(text, aspects):
    doc = TextBlob(text)
    aspect_sentiment = {}
    for aspect in aspects:
        for sentence in doc.sentences:
            if aspect in sentence.string:
                aspect_sentiment[aspect] = sentence.sentiment.polarity
    return aspect_sentiment

df_sampled['aspect_sentiments_base'] = df_sampled.apply(lambda x: aspect_sentiment_base(x['Text'], x['aspects']), axis=1)


# In[ ]:


#df_sampled


# ### Transfomer

# In[ ]:


from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# Prepare data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_review(review):
    return tokenizer.encode_plus(review, truncation=True, padding='max_length', max_length=256)

# Split data
train_reviews, val_reviews = train_test_split(df_sampled, test_size=0.2)

# Tokenize data
train_encodings = train_reviews['Text'].apply(encode_review)
val_encodings = val_reviews['Text'].apply(encode_review)

# Convert to PyTorch dataloaders
train_dataset = torch.utils.data.DataLoader(train_encodings, shuffle=True, batch_size=32)
val_dataset = torch.utils.data.DataLoader(val_encodings, shuffle=False, batch_size=32)

# Initialize model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Training setup
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()


# ### Cross Validation

# In[ ]:



from sklearn.model_selection import cross_val_score

# Function to run cross-validation
def run_cross_validation(model, data, labels):
    scores = cross_val_score(model, data, labels, cv=5)
    return scores

# Run CV
cv_scores = run_cross_validation(model, filtered_df['Text'], filtered_df['Score'])

print(f"Cross-validated scores: {cv_scores}")
print(f"Average CV Score: {cv_scores.mean()}")


# ## Fine-tune BERT

# In[ ]:


# Assuming you have `input_ids`, `attention_mask`, and `labels` prepared for your dataset:

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
model.fit([input_ids, attention_mask], labels, epochs=3, batch_size=8)  # Adjust epochs and batch size as needed

