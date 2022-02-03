# Sentiment Analysis of IMDb Movie Reviews

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#background">Background</a></li>
      </ul>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#nomenclature">Nomenclature</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#research-papers">Research Papers</a></li>
    <li><a href="#contribute">Contribute</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a>
    
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

![SA](https://www.freecodecamp.org/news/content/images/2020/09/wall-5.jpeg)

### Background

For the area of analysis of movie review, sentiment analysis means finding the mood of the public about how they judge a specific movie. And based on the reviews available online, in this instance, IMDb, most movie buffs would love to view the ratings of a movie before they book their tickets, as this may influence their experience from the movie and whether it is worthwhile to spend their money and time. As such, this project aims to build a deep learning algorithm to classify the reviews in the IMDb movie reviews sentiment dataset.

**More information about the methodology and experiments conducted during the course of this project can be found at this [REPORT](https://github.com/patprem/SentimentAnalysis/blob/bcc06d4e5c01555570df185f26e0e45a25fc7366/Report.pdf)**

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

This section lists the major frameworks/libraries and deep learning architectures used to bootstrap this project. 
* [PyTorch](https://pytorch.org/)
* [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/)
* [NLTK: Natural Language Toolkit](https://www.nltk.org/)
* [Matplotlib](https://matplotlib.org/)
* [scikit](https://scikit-learn.org/stable/)
* [NumPy](https://numpy.org/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started
Why is Sentiment Analysis important?
* Humans are subjective creatures and their opinions are important because they reflect their satisfaction with products, services and available technologies. A movie review is an article reflecting its writers’ opinion about a certain movie and criticizing it positively or negatively which enables everyone to understand the overall idea of that movie and make the decision whether to watch it or not, and these reviews can affect the success or failure of a movie. Therefore, a vital challenge is to be able to classify movies reviews to retrieve and analyze watchers more effectively. 
* Movie reviews classification into positive or negative reviews are connected with words occurrences from the reviews text, and whether those words have been used before in a positive or a negative context. These factors help enhance the review understanding process using Sentiment Analysis, where it has become the gateway to understanding consumer needs. Sentiment analysis is concerned with identifying and categorizing opinions which are subjective impressions, not facts but usually expressed in a text and determining whether the writer's feelings, attitudes or emotions towards a particular topic are positive or negative. 
* The aim of this research is to classify movie reviews into positive and negative reviews using Natural Language Processing (NLP) techniques, Linear Support Vector Classifier model and CNN and compare the discrepancies between the two models.

<p align="right">(<a href="#top">back to top</a>)</p>

### Nomenclature
* **Natural Language Processing (NLP)**:
  * **NLTK:** Natural Language ToolKit is a suite of libraries for symbolic and satistical Natural Language Processing (NLP) tasks for English written in Python.
  * **BeautifulSoup:** a Python library used to extract data out of HTML and XML files by eliminating HTML contents/tags such as "<br>" from the movie reviews provided in the datasets.
  * **RegEx (Regular Expressions):** a Python module used to remove other special characters and punctuations, with the exception for upper or lower case letters.
  * **Stopwords** are the English words which does not add much meaning to a sentence, and can safely be ignored without sacrificing the meaning of a sentence. Words like 'he', 'have', 'the' does not provide any insights.
  * **Tokenization:** the process of splitting a string or text into a list of tokens primarily to remove stopwords; first step in NLP projects because it’s the foundation for developing good models and helps better understand the text we have.
  * **Stemming:** a process to extract the base form of the words by removing affixes from the words. Stemming tries to achieve a reduction in words to their root form but the stem itself is not a valid English word.
  * **Lemmatization** has the same objective as Stemming, however, it takes into consideration the morphological analysis of the words, i.e., it ensures that the root word is a valid English word alphabetically and meaningfully.

* **Deep Learning concepts/models:**
  * **Vectorization** (Bag of Words (BoW) model): Word Embeddings, also known as Word Vectorization, is an NLP technique for mapping words or phrases from a lexicon to a corresponding vector of real numbers, which can then be used to derive word predictions and semantics. Vectorization is the process of translating words into numbers.
   1. CountVectorizer
   2. TFIDF Vectorizer 
  * **Linear Support Vector Classifier (LinearSVC)**
  * **Multinomial Naive Bayes**
  * **Convolutional Neural Network (CNN)**

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- PAPERS EXAMPLES -->
## Research Papers
1. Hannah Kim and Young-Seob Jeong, “Sentiment Analysis Using Convolutional Neural Networks”, Applied Sciences, vol. 1, no. 2, pp. 1-3, June 2019. [[link]](https://www.mdpi.com/2076-3417/9/11/2347/pdf-vor)
2. Walaa Medhat, Ahmed Hassan, Hoda Korashy, “Sentiment analysis algorithms and applications: A survey”, Ain Shams University, vol. 5, no. 1093-1113, pp. 2-3, April 2014. [[link]](https://www.researchgate.net/publication/261875740_Sentiment_Analysis_Algorithms_and_Applications_A_Survey)
3. Abdullah Alsaeedi, Mohammad Zubair Khan, “A Study on Sentiment Analysis Techniques of Twitter Data”, International Journal of Advanced Computer Science and Applications (IJACSA), vol. 10, no. 2, pp. 7-8, 2019. [[link]](https://www.researchgate.net/publication/331411860_A_Study_on_Sentiment_Analysis_Techniques_of_Twitter_Data)
4. Munir Ahmad, Shabib Aftab, “Sentiment Analysis using SVM: A Systematic Literature Review”, International Journal of Advanced Computer Science and Applications (IJACSA), vol. 9, no. 2, pp. 3-4, 2018. [[link]](https://thesai.org/Downloads/Volume9No2/Paper_26-Sentiment_Analysis_using_SVM.pdf)
5. Mais Yasen, Sara Tedmori, “Movie Reviews Sentiment Analysis and Classification”, ResearchGate, vol. 1, no. 2, pp. 1-5, April 2019. [[link]](https://www.researchgate.net/publication/332321070_Movies_Reviews_Sentiment_Analysis_and_Classification)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTE -->
## Contribute

If you like this project and interested to contribute:
* Please show your support by ⭐ (star) the project.
* Submit pull requests and improve the repo overall quality.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
This project was mainly inspired from a [Kaggle Competition](https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews) conducted in the year 2020-21. The datasets used in this project borrows from [IMDB Dataset (csv)](https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/data) and [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

<p align="right">(<a href="#top">back to top</a>)</p>
