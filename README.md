# Response Types of Conversational Implicature
This project addresses the linguistic issues that arise in connection with annotating utterances by response types of conversational implicature. The main objectives of this project are to measure ML models' performance of predicting response types of conversational implicature and to analyze which model can outperform for this task.
<br/><br/>

# Introduction
 - Conversational implicature is the core topic of pragmatics, and it is also one of the difficult problems to overcome in natural language processing. However, the research paths of conversational implicatures appear to be contradictory in pragmatics and computer science. In linguistic terms, conversational implicature is the meaning of the speaker’s utterance that is not part of what is explicitly said. In human-computer interactions, the machine fails to understand the implicated meaning unless it is trained with a dataset containing the implicated meaning of an utterance along with the utterance and the context in which it is uttered.
 - In this project, I introduce a small dataset of question-answer pairs with response types, which are 'YES', 'NO', and 'NEUTRAL' (no relations). These response types are the indicators of conversational implicatures.
 - In previous NLP works, there are some papers focused on the dataset for recovering implicatures such as *GRICE: A Grammar-based Dataset for Recovering Implicature and Conversational rEasoning* [[1]](#1) which are not included in the conversation of natural language yet, *Text Classification of Conversational Implicatures Based on Lexical Features* [[2]](#2), which shows that there is a statistical dependence between lexical features and conversational implicatures, and the text classification of implicatures can be performed only based on lexical features, and *Conversational implicatures in English dialogue: Annotated dataset* [[3]](#3). However, there are not many tasks focused on a dataset for recovering conversational implicatures as well as text classification of conversational implicatures, especially in the Thai language. Therefore, measuring ML models' performance with the Thai dialogue dataset has been considered to be an interesting text classification task.
 - The objectives of this project are to measure ML models' performance of predicting response types of conversational implicature and to analyze which model can outperform for this task.
 - The results have shown that the Logistic Regression model with a bag-of-words of both questions and answers performs the best for this task, and the baseline model's performance is still competitive with the pre-trained model, WangchanBERTa. It is probably because of the very small dataset. The future work is to train more models to see more various results and create a larger dataset so that it fits in with many pre-trained models.
<br/><br/>

# Previous Works about Conversational Implicature
 - **GRICE: A Grammar-based Dataset for Recovering Implicature and Conversational rEasoning** [[1]](#1)<br/>
The entire dataset is systematically generated using a hierarchical grammar model, such that each dialogue context has intricate implicatures and is temporally consistent. They further present two tasks, the implicature recovery task followed by the pragmatic reasoning task in conversation, to evaluate the model’s reasoning capability.
 - **Text Classification of Conversational Implicatures Based on Lexical Features** [[2]](#2)<br/>
Main work of this paper includes: First, based on 600 corpora in the annotated dataset, the values of 20 lexical features of each corpus are obtained by automatic calculation. Second, meta-transformer of logistic regression for selecting features is adopted for feature selection and ranking. Third, after determining the features, the text is classified by the binomial logistic regression with the type of implicatures as labels. Fourth, results are tested for significance to identify relationships between variables. Experiments show that there is a statistical dependence between lexical features and conversational implicatures, and the text classification of implicatures can be performed only based on lexical features.
 - **Conversational implicatures in English dialogue: Annotated dataset** [[3]](#3)<br/>
In this paper, they introduce a dataset of dialogue snippets with three constituents, which are the context, the utterance, and the implicated meanings. These implicated meanings are the conversational implicatures. The utterances are collected by transcribing from listening comprehension sections of English tests like TOEFL as well as scraping dialogues from movie scripts available on IMSDb (Internet Movie Script Database). The utterances are manually annotated with implicatures.
<br/><br/>

# Approach / Methodology
 - Created a dialogue implicature dataset which includes question utterances manually collected from [TNC (Thai National Corpus)](https://www.arts.chula.ac.th/ling/tnc3/), and answer utterances manually generated.
 - Then annotated them manually with a definite/probable 'YES', 'NO', or 'NEUTRAL' for 300 indirect polar questions according to annotation guidelines.
 - Trained models and tested ML models' performance with this dataset.
 - Compared and analyzed the results.
<br/><br/>

# Models
**The baseline model**<br/>
Logistic regression [[4]](#4) is a statistical model that uses Logistic function to model the conditional probability. It is commonly used for prediction and classification problems. In this project, this model was trained in two ways.
 - Logistic Regression: bag-of-words (only answers) [unigram]
 - Logistic Regression: bag-of-words (both questions and answers) [bigram]

**The pre-trained model**<br/>
WangchanBERTa [[5]](#5) is a Thai language model based on RoBERTa-base architecture on a large, deduplicated, cleaned training set (78GB in total size), curated from diverse domains of social media posts, news articles and other publicly available datasets. This model outperforms strong baselines (NBSVM, CRF and ULMFit) and multi-lingual models (XLMR and mBERT) on both sequence classification and token classification tasks in human-annotated, mono-lingual contexts. In this project, this model was trained in two ways.
 - WangchanBERTa (without hyperparameter tuning)
 - WangchanBERTa (with hyperparameter tuning)
<br/><br/>

# Dataset
 - [Annotation guidelines](https://acrobat.adobe.com/link/review?uri=urn:aaid:scds:US:32f73871-7744-3a29-ace0-11ca0fca99a7)
 - [Raw dataset](https://docs.google.com/spreadsheets/d/1Ji2k0cT5RLNC6C2xYeZczRuN35PEP2QgbuUtwDdOB9o/edit?usp=sharing)
 - There are three parts of the dataset; 'question', 'answer', and 'label'.
 - It includes 300 question-answer pairs with three labels; 'YES', 'NO', and 'NEUTRAL'
<br/><br/>

| Label | Frequency | Percentage |
|-------| --------- | ---------- |
| YES | 102 | 34% |
| NO | 101 | 33.67% |
| NEUTRAL | 97 | 32.33% |

Dataset examples:<br/>
![image](https://user-images.githubusercontent.com/40376515/204923565-1395c530-cec3-49c5-a218-718a4d9f1668.png)
<br/><br/>

# Experiment Setup
All processes of implementation were developed on **Google Colab** (free version).
<br/><br/>
**Computer specs**<br/>
Processor: 11th Gen Intel(R) Core(TM) i7-11370H @ 3.30GHz   3.00 GHz<br/>
Installed RAM: 24.0 GB (23.7 GB usable)<br/>
<br/>
Dataset: total 300 question-answer pairs with three labels<br/>
The dataset was split into train set: 240 pairs (80%), dev set: 30 pairs (10%), and test set: 30 pairs (10%)<br/>

<ins>Baseline model: Logistic Regression</ins><br/>

**Model 1 Bag-of-words (only answers) [unigram]**<br/>
 - Created a bag-of-words from only answer utterances
 - Tokenizer: 'uttacut' from the library 'pythainlp'
 - Training with DictVectorizer
 - Simple 'LogisticRegression()'

**Model 2 Bag-of-words (questions and answers) [bigram]**<br/>
 - Created a bag-of-words from both question and answer utterances
 - Tokenizer: 'uttacut' from the library 'pythainlp'
 - Removing stop words
 - Training with CountVectorizer(ngram_range=(2, 2))
 - LogisticRegression(C=0.1, dual=True, solver='liblinear', max_iter=10000)

<ins>Pre-trained model: WangchanBERTa</ins><br/>

**Model 3 WangchanBERTa (without hyperparameter tuning)**<br/>
 - Tokenizer: 'wangchanberta-base-att-spm-uncased'
 - Format: [CLS] 'question' [SEP] 'answer' [SEP]
 - Training with pre-trained settings

**Model 4 WangchanBERTa (with hyperparameter tuning)**<br/>
 - Tokenizer: 'wangchanberta-base-att-spm-uncased'
 - Format: [CLS] 'question' [SEP] 'answer' [SEP]
 - Epochs: 4
 - Learning rate: 2e-5
 - Training batch size: 16
 - Testing batch size: 8
<br/><br/>

# Results
The results have shown that the Logistic Regression model with a bag-of-words of both questions and answers performs the best for this task, and the baseline model's performance is still competitive with the pre-trained model, WangchanBERTa. For comparison within the same model, the WangchanBERTa without hyperparameter tuning outperforms the WangchanBERTa with hyperparameter tuning. It is probably because of a very small dataset, and the test set has only 30 pairs. Therefore, it cannot conclude that the WangchanBERTa model does not perform well on this task. In addition, I have trained the cross-product neural network model with this dataset, but there are some problems with vectors and word indices. So, I cannot inform this model's performance. The future work is to train more models to see more various results and create a larger dataset so that it fits in with other pre-trained models.<br/><br/>
**Model 1 Bag-of-words (only answers) [unigram]** [[6]](#6)<br/>
![image](https://user-images.githubusercontent.com/40376515/206033187-e3668b03-3d5b-480e-9746-7d437890b66a.png)<br/><br/>
**Model 2 Bag-of-words (questions and answers) [bigram]** [[6]](#6)<br/>
![image](https://user-images.githubusercontent.com/40376515/206033329-eaf372ae-b534-4ee7-b97d-268dd3fa65c3.png)<br/><br/>
**Model 3 WangchanBERTa (without hyperparameter tuning)** [[6]](#6)<br/>
![image](https://user-images.githubusercontent.com/40376515/206033367-c38fc238-3267-4893-b036-87e8a77a9960.png)<br/><br/>
 **Model 4 WangchanBERTa (with hyperparameter tuning)** [[6]](#6)<br/>
![image](https://user-images.githubusercontent.com/40376515/206033418-db02a448-3bb4-4014-9bbe-3819ac80f715.png)
<br/>
<a id="6">[6]</a>
Label '0' is 'NEUTRAL', label '1' is 'YES', and label '2' is 'NO'.<br/>

# Model Comparison
| Model | Precision | Recall | Accuracy | F1-score |
| ----- | --------- | ------ | -------- | -------- |
| Bag-of-words (only answers) | 0.47 | 0.50 | 0.47 | 0.48 |
| Bag-of-words (questions and answers) | 0.53 | 0.57 | 0.53 | 0.54 |
| WangchanBERTa (without hyperparameter tuning) | 0.48 | 0.47 | 0.47 | 0.47 |
| WangchanBERTa (with hyperparameter tuning) | 0.11 | 0.33 | 0.33 | 0.17 |
<br/>

# Conclusions
In this project, I present my approach to collect and annotate a small dataset of dialogues with implicatures associated with the response utterance. The collected dataset can be used as a reference for identifying and synthesising conversational implicatures. The question utterances are manually collected from [TNC (Thai National Corpus)](https://www.arts.chula.ac.th/ling/tnc3/), and the answer utterances are manually generated by myself. As implicatures are generated in a wide range of situations and are highly dependent on the hearer's understanding, I have primarily focused on the polar questions where an indirect answer without an explicit 'YES' or 'NO' generates implicatures, and questions with unrelated answers as well.
The results from models have shown that the Logistic Regression model with a bag-of-words of both questions and answers performs the best for this task, and the baseline model's performance is still competitive with the pre-trained model, WangchanBERTa. For comparison within the same model, the WangchanBERTa without hyperparameter tuning outperforms the WangchanBERTa with hyperparameter tuning.
In the future, I am planning to add more question-answer pairs in various contexts and train other models to see more results.
<br/><br/>

# References
<a id="1">[1]</a>
Zilong Zheng, Shuwen Qiu, Lifeng Fan, Yixin Zhu, Song-Chun Zhu, 
"GRICE: A Grammar-based Dataset for Recovering Implicature and Conversational rEasoning," 
*Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, August 1–6, (2021)*: 2074–2085.<br/>
<a id="2">[2]</a>
Xianbo Li. (2022). "Text Classification of Conversational Implicatures Based on Lexical Features,"
*Applied Artificial Intelligence*, 36(1): 3160-3175, 
DOI: [10.1080/08839514.2022.2127598](https://doi.org/10.1080/08839514.2022.2127598)<br/>
<a id="3">[3]</a>
Elizabeth Jasmi George, Radhika Mamidi. (2020). "Conversational implicatures in English dialogue: Annotated dataset," 
*Procedia Computer Science*, 171(9): 2316-2323, 
DOI: [10.1016/j.procs.2020.04.251](https://www.researchgate.net/publication/341907785_Conversational_implicatures_in_English_dialogue_Annotated_dataset)<br/>
<a id="4">[4]</a>
Vimal, Bhartendoo. (2020). Application of Logistic Regression in Natural Language Processing. *International Journal of Engineering Research and Technical Research*. V9. DOI: [10.17577/IJERTV9IS060095](https://www.researchgate.net/publication/342075482_Application_of_Logistic_Regression_in_Natural_Language_Processing)<br/>
<a id="5">[5]</a>
Lowphansirikul, L., Polpanumas, C., Jantrakulchai, N., & Nutanong, S. (2021). Wangchanberta: Pretraining transformer-based thai language models. *[arXiv preprint arXiv:2101.09635](https://arxiv.org/abs/2101.09635)*.
