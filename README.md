# Response Types of Conversational Implicature
This project addresses the linguistic issues that arise in connection with annotating utterances by response types of conversational implicature.
<br/><br/>

# Introduction
 - Motivation: Why is this task interesting or important?
 - What other people have done? What model have previous work have used? What did they miss? (1-2 paragraphs)

Previous works about conversational implicature:
 - work 1: GRICE
 - work 2:
 - work 3:


 - Summarize your idea

The objective of this project is to measure ML models' performance in predicting response types of conversational implicature and to analyze whether the models can understand the real context of conversations or not. It is reckoned as one of the most interesting text classification task.
 - Summarize your results

The results have shown that ...
<br/><br/>

# Approach / Methodology

<br/><br/>

# Models
The baseline models:
 - Logistic Regression: bag-of-words (only answers) [unigram]
 - Logistic Regression: bag-of-words (questions and answers) [bigram]

The pre-trained models:
 - BERT
 - WangchanBERTa

Explain your model here how it works.
 - Input is ...
 - Output is ...
 - Model descriptions
 - Equation (if necessary)
<br/><br/>

# Dataset
 - [Annotation guidelines](https://acrobat.adobe.com/link/review?uri=urn:aaid:scds:US:32f73871-7744-3a29-ace0-11ca0fca99a7)
 - [Raw dataset](https://docs.google.com/spreadsheets/d/1Ji2k0cT5RLNC6C2xYeZczRuN35PEP2QgbuUtwDdOB9o/edit?usp=sharing)
 - There are three parts of the dataset; "question", "answer", and "label".
 - 300 question-answer pairs with three labels
<br/><br/>

| Label | Frequency | Percentage |
|-------| --------- | ---------- |
| YES | 102 | 34% |
| NO | 101 | 33.67% |
| NEUTRAL | 97 | 32.33% |

<br/><br/>

# Experiment setup
All processes of implementation were developed on Google Colab (free version).

Baseline model: Logistic Regression<br/>

Model 1 Bag-of-words (only answers) [unigram]<br/>
 Preprocessing:
 - Tokenizer: "uttacut" from library "pythainlp"
 Training:
 - Using DictVectorizer(sparse=True)
 - Using simple "LogisticRegression()"

Model 2 Bag-of-words (questions and answers) [bigram]<br/>
 Preprocessing:
 - Tokenizer: "uttacut" from library "pythainlp"
 - Removing stop words
 Training:
 - Using "CountVectorizer(ngram_range=(2, 2), max_features=800000)"
 - Using "LogisticRegression(C=0.1, dual=True, solver='liblinear', max_iter=10000)"

The pre-trained models: BERT, WangchanBERTa<br/>

Model 3 BERT (pre-trained model)<br/>
 Preprocessing:
 - Tokenizer: "bert-base-uncased"
 - 
 - Which pre-trained model? How did you pretrain embeddings?
 - Computer spec & how long?
 - Hyperparameter tuning? Dropout? How many epochs?
<br/><br/>

# Results
 - How did it go? + Interpret results
<br/><br/>

# Model comparison
| Model | Accuracy |
| -------- | --------- |
| LR | 67% |
| BERT | 86% |

<br/><br/>

# Conclusions

