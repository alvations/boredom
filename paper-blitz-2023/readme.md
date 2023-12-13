# Doing a paper-blitz for 2023 ACL + EMNLP papers

It starts with this tweet: https://twitter.com/alvat# Sions/status/1734781934998577198

> I'm at it again. Doing an #nlproc "paper blitz" for @aclmeeting and @emnlpmeeting 2023 papers.
> 
> Let's try live tweeting while I blitz 😁


# Step 0: Judge the paper by its title

From https://twitter.com/alvations/status/1734783858653147262

> There are soooo many paper this year. 
>
> The only sane way around it is to scope down my #nlproc interest from #neuralempty to general "evaluation", cos these days that is what I work on, and so I literally went to 
@aclanthology and did CTR+F "evalua":


**Time taken:** 20 mins

**Notes:** I didn't restrict myself to "evaluation" papers but when I see something interesting around the evaluation papers, I selected them for blitzing. 

## Papers selected

```
https://aclanthology.org/2023.acl-long.730/
https://aclanthology.org/2023.acl-long.821/
https://aclanthology.org/2023.acl-long.866/
https://aclanthology.org/2023.acl-long.870/
https://aclanthology.org/2023.acl-demo.30/
https://aclanthology.org/2023.acl-demo.56/
https://aclanthology.org/2023.acl-industry.36/
https://aclanthology.org/2023.acl-industry.40/
https://aclanthology.org/2023.emnlp-main.135/
https://aclanthology.org/2023.emnlp-main.258/
https://aclanthology.org/2023.emnlp-main.308/
https://aclanthology.org/2023.emnlp-main.390/
https://aclanthology.org/2023.emnlp-main.464/
https://aclanthology.org/2023.emnlp-main.496/
https://aclanthology.org/2023.emnlp-main.584/
https://aclanthology.org/2023.emnlp-main.585/
https://aclanthology.org/2023.emnlp-main.593/
https://aclanthology.org/2023.emnlp-main.676/
https://aclanthology.org/2023.emnlp-main.699/
https://aclanthology.org/2023.emnlp-main.724/
https://aclanthology.org/2023.emnlp-main.859/
https://aclanthology.org/2023.emnlp-main.859/
https://aclanthology.org/2023.findings-emnlp.58/
https://aclanthology.org/2023.findings-emnlp.264/
https://aclanthology.org/2023.findings-emnlp.278/
https://aclanthology.org/2023.findings-emnlp.722/
https://aclanthology.org/2023.findings-emnlp.966/
https://aclanthology.org/2023.findings-emnlp.1001/
https://aclanthology.org/2023.calcs-1.1/
https://aclanthology.org/2023.wmt-1.49/
https://aclanthology.org/2023.wmt-1.61/
https://aclanthology.org/2023.wmt-1.67/
https://aclanthology.org/2023.wmt-1.96/
https://aclanthology.org/2023.wmt-1.99/
https://aclanthology.org/2023.wmt-1.100/
https://aclanthology.org/2023.wmt-1.97/
https://aclanthology.org/2023.wmt-1.95/
https://aclanthology.org/2023.wmt-1.80/
```


# Step 1: Categorize the papers

The goal here is to categorize the papers selected within 25-30 mins.

I'm getting too lazy to copy+pasting the title from the URLs I've copied, so I did this =) 

```python
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

selected = """https://aclanthology.org/2023.acl-long.730/
https://aclanthology.org/2023.acl-long.821/
https://aclanthology.org/2023.acl-long.866/
..."""

titles = []

for url in tqdm(selected.split('\n')):
    response = requests.get(url)
    bsoup = BeautifulSoup(response.content.decode('utf8'))
    titles.append(bsoup.find('title').text.rpartition('-')[0])
```

Then I got even lazier, why categorize when I can make a model do it for me and I edit it afterwards.

From https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/fast_clustering.py

```python
from sentence_transformers import SentenceTransformer, util
import os
import csv
import time

# Model for computing sentence embeddings. We use one trained for similar questions detection
model = SentenceTransformer('all-MiniLM-L6-v2')

corpus_embeddings = model.encode(
    [t.lower() for t in titles], batch_size=15, 
    show_progress_bar=True, convert_to_tensor=True)

#Two parameters to tune:
#min_cluster_size: Only consider cluster that have at least 2 elements
#threshold: Consider sentence pairs with a cosine-similarity larger than threshold of 0.5
clusters = util.community_detection(corpus_embeddings, min_community_size=2, threshold=0.5)

#Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    for sentence_id in cluster:
        print(titles[sentence_id])
    print('-------')
```

[out]:

```
Are Human Explanations Always Helpful? Towards Objective Evaluation of Human Natural Language Explanations 
Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators 
ParroT: Translating during Chat using Large Language Models tuned with Human Translation and Feedback 
Can Large Language Models Be an Alternative to Human Evaluations? 
Large Language Models are Not Yet Human-Level Evaluators for Abstractive Summarization 
MEGA: Multilingual Evaluation of Generative AI 
Extrinsic Evaluation of Machine Translation Metrics 
EpiK-Eval: Evaluation for Language Models as Epistemic Models 
The Devil Is in the Errors: Leveraging Large Language Models for Fine-grained Machine Translation Evaluation 
HyperT5: Towards Compute-Efficient Korean Language Modeling 
Comparing the Evaluation and Production of Loophole Behavior in Humans and Large Language Models 
Evaluating Evaluation Metrics: A Framework for Analyzing NLG Evaluation Metrics using Measurement Theory 
Training and Meta-Evaluating Machine Translation Evaluation Metrics at the Paragraph Level 
-------
eBLEU: Unexpectedly Good Machine Translation Evaluation Using Simple Word Embeddings 
The OPUS-MT Dashboard – A Toolkit for a Systematic Evaluation of Open Machine Translation Models 
Terminology-Aware Translation with Constrained Decoding and Large Language Model Prompting 
Automating Behavioral Testing in Machine Translation 
Trained MT Metrics Learn to Cope with Machine-translated References 
A Benchmark for Evaluating Machine Translation Metrics on Dialects without Standard Orthography 
Towards Better Evaluation for Formality-Controlled English-Japanese Machine Translation 
-------
Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks 
NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark 
```

**Talking to myself:** Not bad, I get 3 groups automatically, let me see if I can refine the groups and add some categories title to them.


# Step 1b: Clean up the clusters and label them.


### Humans + Evaluation
- Are Human Explanations Always Helpful? Towards Objective Evaluation of Human Natural Language Explanations 
- Can Large Language Models Be an Alternative to Human Evaluations? 
- Large Language Models are Not Yet Human-Level Evaluators for Abstractive Summarization
- ParroT: Translating during Chat using Large Language Models tuned with Human Translation and Feedback
- Comparing the Evaluation and Production of Loophole Behavior in Humans and Large Language Models 

### General Evaluation
- MEGA: Multilingual Evaluation of Generative AI
- Beyond Factuality: A Comprehensive Evaluation of Large Language Models as Knowledge Generators
- EpiK-Eval: Evaluation for Language Models as Epistemic Models 

### Meta-evaluation
- Evaluating Evaluation Metrics: A Framework for Analyzing NLG Evaluation Metrics using Measurement Theory
- Training and Meta-Evaluating Machine Translation Evaluation Metrics at the Paragraph Level 

### Machine Translation metrics related
- Extrinsic Evaluation of Machine Translation Metrics
- The Devil Is in the Errors: Leveraging Large Language Models for Fine-grained Machine Translation Evaluation 
- eBLEU: Unexpectedly Good Machine Translation Evaluation Using Simple Word Embeddings 
- The OPUS-MT Dashboard – A Toolkit for a Systematic Evaluation of Open Machine Translation Models
- Automating Behavioral Testing in Machine Translation
- Trained MT Metrics Learn to Cope with Machine-translated References
- A Benchmark for Evaluating Machine Translation Metrics on Dialects without Standard Orthography 
- Towards Better Evaluation for Formality-Controlled English-Japanese Machine Translation

### Machine Translation / Model buidling
- Terminology-Aware Translation with Constrained Decoding and Large Language Model Prompting 
- HyperT5: Towards Compute-Efficient Korean Language Modeling 
 
### Data Contamination
- Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks 
- NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark 

**Time taken:** ~15 mins for the code and cluster and playing around the `min_community_size` and `threshold` and ~5 mins to just go through the clusters quickly, reshuffle some papers and put labels on them. Amazing!! Clustering did help with the categorization step of "paper-blitz"!!


# The actual work... Step 2: Reading the papers

We have 37 papers and 6 groups of paper, and 2 hours before I have to do serious work at work... 
That goes down to ~3 mins per paper.

- **Humans + Evaluation**: 5 papers = 15 mins
- **General Evaluation**: 3 papers = 9 mins
- **Meta-evaluation**: 2 papers = 6 mins
- **MT metrics**: 8 papers = 24 mins
- **MT / LM modeling** 2 papers = 6 mins
- **Data contamination** 2 papers = 6 mins

**Me talking to myself:** 37 papers in 2 hours... 頑張れ!


----


### Are Human Explanations Always Helpful? Towards Objective Evaluation of Human Natural Language Explanations 

Problem: Can we evaluate human alignment/explanation data by measuring their helpfulness towards model predictions?

Approach: 
  - An extension of the "Simulatability" metric from  https://arxiv.org/abs/1702.08608
  - a prompt-based unified data format that can convert classification or multiple choice tasks into a generation task in/output
      - **Me commenting to myself**: okay, so just make everything a nail so that your hammer works, not that I disagree with it but we did the same on  https://arxiv.org/abs/1812.05774 =)
   
Strength:
  - Valid work to propose a metric and look at how we evaluate data's usefulness w.r.t. to the model

Weakness (nitpicking):
  - There are sooooo many bolded and/or colored text in the paper @_@

Usefulness:
  - Good to know kind of paper, no particular resource created/provided...

**Time taken:** 4 mins

---

### Can Large Language Models Be an Alternative to Human Evaluations?

Problem: Can we just eliminate humans to evaluate models?

Approach: This picture sums it up

<img src="https://github.com/alvations/boredom/assets/1050316/5cd6cee5-8a61-4d74-a5ed-0b96003ea70d" width="220">

Results: 
 - Use LLM to do eval cos humans disagree with each other models don't
 - Use LLM cos they don't have prior knowledge or get primed by general sample / model behaviors
 - Use LLM cos its cheaper, (**Me**: Okay @_@)

Strength:
 - This work needs to exists so that we can cite and use it to qualify us to try replacing humans (slowly)


Weakness:
 - Don't quite agree that humans get primed but not LLM, because
     - we can randomized the data such that it's blind to the human evaluators
     - we can also try priming LLM with longer context to see if they can get primed/tricked like humans (Good idea for a fun paper!!)

Usefulness:
 - Again, another good to know paper, no resource provided


Time taken: 7 mins


----

## ParroT: Translating during Chat using Large Language Models tuned with Human Translation and Feedback

Problem: What if we can prompt the model to make it do different things?

Approach: This picture sums it up

<img src="https://github.com/alvations/boredom/assets/1050316/40f4cc69-6f84-4d0d-9ee3-4090f579ba14" width="220">

Strength:
 - There's a lot of nitty gritty details on how the experimental setup provides one hypothesis after another.
 - This might be the right direction for #neuralempty, to start with the MT model, then make it LLM-like capabilities. Not the reverse where there's some "emergent abilities" and we force it to scale/learn multilinguality

Weakness:
 - While it's out-of-scope for the paper, the open source/public alternatives for MT models (not LLM) were not compared to. I guess the commercial APIs is kinda representative for the sake of understanding how far the LLM-ish approach to MT, but it'll still be nice to see just quote/compare Table 3 results to vanilla transformers

Usefulness:
 - the prompts, we can re-use them to make LLMs do more than "GEMBA-like" work.
 - Generating "Error-Guided" translations as synthetic data for tuning QE metric might be a good idea
 - Code and model on https://github.com/wxjiao/ParroT

Time taken: 9 mins

----

## Large Language Models are Not Yet Human-Level Evaluators for Abstractive Summarization

Problem: How good are LLMs to replace humans evals for summarization?

Approach: 

1) Prompt it out (Reason then Score):

<img src="https://github.com/alvations/boredom/assets/1050316/0493ba95-eefd-4291-8e15-772b1975d3b6" width="220">


2) Prompt it out T5 style (Give it some MCQ choices)

<img src="https://github.com/alvations/boredom/assets/1050316/bd387635-dc28-48ac-ad82-194fc07f9755" width="220">


Strength:
 - A comprehensive study across ROUGE, BERTScore, BartScore and Prompt-GPT scores

Weakness:
 - From gisting the paper, nothing much. Good work =)

Usefulness:
 - The Reason then Score style should be applied #neuralempty QE, actually some of the public models like Anthropic's Claude is already giving the reasoning after the score when you GEMBA-prompt it.
   - More exploration to this "Reason then Score" idea is good for MT eval
 - Code and data on https://github.com/DAMO-NLP-SG/LLM_summeval

----

## Comparing the Evaluation and Production of Loophole Behavior in Humans and Large Language Models

Okay this paper, hmmmm, first when I google I found it on ICML https://icml.cc/virtual/2023/27884 and then seeing it on https://aclanthology.org/2023.findings-emnlp.264/ 

Hmmmm, hmmm, okay disqualify it as not part of ACL/EMNLP and I can skip blitzing this (but I'll come back to this paper after my blitz session).


