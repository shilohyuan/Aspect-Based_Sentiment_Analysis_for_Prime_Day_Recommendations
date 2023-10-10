# Strategic Showcase: Leveraging Unsupervised Aspect Sentiment Analysis for Optimal Prime Day Product Selection

### Problem Statement  
In today's online retail environment, events like Prime Day create significant business value. Optimizing product recommendations for these events can drive better user experience and increase sales. Through the lens of unsupervised aspect-based sentiment analysis, this project aims to curate products that resonate best with customer sentiments, particularly concerning their price value perceptions.   

**Keywords**: Sentiment Analysis, Aspect-Based Analysis, Unsupervised Learning, Prime Day Optimization, Strategic 
![WordCloud](download.png)

### Data Preprocessing  
The provided dataset encompasses > 300 thousands text reviews detailing user feedback on various products. To facilitate efficient analysis:

Standardized the text length and structure for uniformity.
Tokenized and encoded the reviews for model consumption.
Divided the dataset into an 80% training set and a 20% testing set.

### Methodology
Instead of the conventional transfer learning, our approach leans on state-of-the-art transformer models known for their proficiency in understanding context and nuances in text data.

Model 1: Basic BERT model with a sentiment classification head.
Model 2: Fine-tuned BERT for aspect-based sentiment detection.
