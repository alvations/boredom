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

```
from sentence_transformers import SentenceTransformer, util
import os
import csv
import time


# Model for computing sentence embeddings. We use one trained for similar questions detection
model = SentenceTransformer('all-MiniLM-L6-v2')

corpus_embeddings = model.encode(
    titles, batch_size=15, 
    show_progress_bar=True, convert_to_tensor=True)

#Two parameters to tune:
#min_cluster_size: Only consider cluster that have at least 25 elements
#threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(corpus_embeddings, min_community_size=2, threshold=0.75)
```
