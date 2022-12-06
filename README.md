# Response Types of Conversational Implicature
This project addresses the linguistic issues that arise in connection with annotating utterances by response types of conversational implicature.
<br/><br/>

# Introduction
 - Conversational implicature is the core topic of pragmatics, and it is also one of the difficult problems to be overcome in natural language processing. However, the research paths of conversational implicatures appear to be contradictory in pragmatics and computer science. In linguistic term, conversational implicature is the meaning of the speaker’s utterance that is not part of what is explicitly said. In human-computer interactions, the machine fails to understand the implicated meaning unless it is trained with a dataset containing the implicated meaning of an utterance along with the utterance and the context in which it is uttered.
 - In this project, I introduce a small dataset of question-answer pairs with response types, which are YES, NO, and NEUTRAL (no relations). These response types are the indicators of conversational implicatures.
 - In previous NLP works, there are some papers focused on dataset for recovering implicatures such as *GRICE: A Grammar-based Dataset for Recovering Implicature and Conversational rEasoning* [[1]](#1) which are not included the conversation of natural language yet, *Text Classification of Conversational Implicatures Based on Lexical Features* [[2]](#2), which shows that there is a statistical dependence between lexical features and conversational implicatures, and the text classification of implicatures can be performed only based on lexical features, and *Conversational implicatures in English dialogue: Annotated dataset* [[3]](#3). However, there are not many tasks focused on dataset for recovering conversational implicatures as well as text classification of conversational implicatures especially in Thai language. Therefore, measuring ML models' performance with Thai dialogue dataset has been reckoned as an interesting text classification task.
 - The objective of this project is to measure ML models' performance in predicting response types of conversational implicature and to analyze whether the models can understand the real context of conversations.
 - The results have shown that ...
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
 - Then annotated them manually with a definite/probable 'YES', 'NO', or 'NEUTRAL' for 300 indirect polar questions.
 - Trained and tested ML models' performance with this dataset.
 - Compared and analyzed the results.
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
 - There are three parts of the dataset; "QUESTION", "ANSWER", and "LABEL".
 - 300 question-answer pairs with three labels
<br/><br/>

| Label | Frequency | Percentage |
|-------| --------- | ---------- |
| YES | 102 | 34% |
| NO | 101 | 33.67% |
| NEUTRAL | 97 | 32.33% |

Dataset examples:<br/>
![image](https://user-images.githubusercontent.com/40376515/204923565-1395c530-cec3-49c5-a218-718a4d9f1668.png)
<br/><br/>

# Experiment setup
All processes of implementation were developed on Google Colab (free version).

Baseline model: Logistic Regression<br/>

Model 1 Bag-of-words (only answers) [unigram]<br/>
 Preprocessing:
 - Tokenizer: "uttacut" from library "pythainlp"<br/>
 Training:
 - Using DictVectorizer(sparse=True)
 - Using simple "LogisticRegression()"

Model 2 Bag-of-words (questions and answers) [bigram]<br/>
 Preprocessing:
 - Tokenizer: "uttacut" from library "pythainlp"
 - Removing stop words<br/>
 Training:
 - Using "CountVectorizer(ngram_range=(2, 2), max_features=800000)"
 - Using "LogisticRegression(C=0.1, dual=True, solver='liblinear', max_iter=10000)"

The pre-trained models: BERT, WangchanBERTa<br/>

Model 3 BERT<br/>
 Preprocessing:
 - Tokenizer: "bert-base-uncased"<br/>
 Training:
 - Epochs:
 - Metric: accuracy<br/>
 Hyperparameter tuning:
 - Learning rate:
 - Training batch size:
 - Testing batch size: __ <br/>
 
 Model 4 WangchanBERTa<br/>
 Preprocessing:
 - Tokenizer: "wangchanberta-base-att-spm-uncased"<br/>
 Training:
 - Epochs:
 - Metric: accuracy<br/>
 Hyperparameter tuning:
 - Learning rate:
 - Training batch size:
 - Testing batch size:
<br/><br/>

# Results
 - How did it go? + Interpret results
<br/><br/>

# Model comparison
| Model | F1-score |
| -------- | --------- |
| Bag-of-words (only answers) | 47% |
| Bag-of-words (questions and answers) | 53% |
| BERT | --% |
| WangchanBERTa | --% |

<br/><br/>

# Conclusions

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
