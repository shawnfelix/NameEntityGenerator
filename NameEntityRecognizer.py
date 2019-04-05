#python includes
import sys
import json
import copy
import random
from pprint import pprint
from collections import Counter
import re

#twitter api
import tweepy

#standard probability includes:
import numpy as np #matrices and data structures

#scikit learn imports
import scipy.stats as ss #standard statistical operations
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

consumer_key = '#INSERT TWITTER API KEY HERE#'
consumer_secret = '#INSERT TWITTER API KEY HERE#'

twitter_access_token = "#INSERT TWITTER API KEY HERE#"
twitter_access_token_secret = "#INSERT TWITTER API KEY HERE#"

#tweepy custom streamlistener for getting tweets
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.tweet_limit = 1000
        self.tweets = []
        self.file_name = "tweets.txt"
    def on_status(self, status):
        tweet = status._json
        with open(self.file_name, 'a') as file:
            file.write(json.dumps(tweet["text"]) + '\n')
            file.close()
        self.tweets.append(status.text)
        self.num_tweets += 1
        if self.num_tweets < self.tweet_limit:
            return True
        else:
            return False
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False

class NgramModel(object):
    pass

#tokenize the words for given string
def tokenize(sent):
    wordRE = re.compile(r'((?:[A-Z]\.)+|(?:[\.,!?;"])|(?:(?:\#|\@)?[A-Za-z0-9_\-]+(?:\'[a-z]{1,3})?))', re.UNICODE)

    #input: a single sentence as a string.
    #output: a list of each “word” in the text
   
    tokens = wordRE.findall(sent)
    return tokens

def getFeaturesForTarget(tokens, targetI, wordToIndex):
    #input: tokens: a list of tokens, 
    #       targetI: index for the target token
    #       wordToIndex: dict mapping ‘word’ to an index in the feature list. 
    #output: list (or np.array) of k feature values for the given target
    
    #is the word capitalized
    wordCap = np.array([1 if tokens[targetI][0].isupper() else 0])
    oovIndex = len(wordToIndex)
    
    #first letter of the target word
    letterVec = np.zeros(257)
    val = ord(tokens[targetI][0])
    if val < 256:
        letterVec[val] = 1
    else:
        letterVec[256] = 1
        
    #length of the word:
    length = np.array([len(tokens[targetI])])
    
    #previousWord:
    prevVec = np.zeros(len(wordToIndex)+1)#+1 for OOV
    if targetI > 0:
        try:
            prevVec[wordToIndex[tokens[targetI - 1]]] = 1
        except KeyError:
            prevVec[oovIndex] = 1
            pass#no features added

    #targetWord:
    targetVec = np.zeros(len(wordToIndex)+1)
    try:
        targetVec[wordToIndex[tokens[targetI]]] = 1
    except KeyError:
        targetVec[oovIndex] = 1
        #print("unable to find wordIndex for '", tokens[targetI], "' skipping")
        pass
    
    #nextWord
    nextVec = np.zeros(len(wordToIndex)+1)
    if targetI+1 < len(tokens) :
        try:
            nextVec[wordToIndex[tokens[targetI + 1]]] = 1
        except KeyError:
            nextVec[oovIndex] = 1
            pass
        
    featureVector = np.concatenate((wordCap, letterVec, length, prevVec,\
                                    targetVec, nextVec))
    
    return featureVector

def NamedEntityGenerativeSummary(named_entity, 
                        twitter_access_token, twitter_access_token_secret):
    #get cap.1000 conll common nouns words
    commonNouns = getConllList('cap.1000.conll')
    #file name for training model
    corpus = 'daily547.conll'
    corpus2 = 'oct27.conll'

    ########################################################
    #                   PART 1
    #train the named entity recognizer; save it to an object
    ########################################################
    wordToIndex = set()
    wordToIndexTwice = set()
    tagToNum = set()
    taggedSents = getConllTags(corpus)
    #taggedSents2 = getConllTags(corpus2)
    #taggedSents = taggedSents + taggedSents2

    #--------------------------------------------------------
    #train NER 
    c = 0
    for sent in taggedSents:
        if sent: 
            words, tags = zip(*sent)
            if c > 0:
                #check if new set of words in wordToIndex - union to wordToIndexTwice if already appears
                wordToIndexTwice |= wordToIndex.intersection(set(words))
            c += 1
            wordToIndex |= set(words) #union of the words into the set
            
            tagToNum |= set(tags) #union of all the tags into the set
    print("[Read ", len(taggedSents), " Sentences]")
    #make dictionaries for converting words to index and tags to ids:
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)} 
    wordToIndexTwice = {w: i for i, w in enumerate(wordToIndexTwice)} 
    numToTag = list(tagToNum) #mapping index to tag
    tagToNum = {numToTag[i]: i for i in range(len(numToTag))}

    #--------------------------------------------------------------------------------
    #2b) Call feature extraction on each target
    X = []
    y = []

    print("[Extracting Features]")
    for sent in taggedSents:
        if sent: 
            words, tags = zip(*sent)
            for i in range(len(words)):
                y.append(1 if tags[i] == '^' else 0) #append y with class label
                X.append(getFeaturesForNER(words, i, wordToIndex, wordToIndexTwice, commonNouns))

    X, y = np.array(X), np.array(y)
    print("[Done X is ", X.shape, " y is ", y.shape, "]")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30,
                                                    random_state=42)
    print("[Broke into training/test. X_train is ", X_train.shape, "]")

    #-------------------------------------------------------------------------
    #3 Train the model. 
    print("[Training the model]")
    tagger = trainTagger(X_train, y_train)
    print("[done]")

    #4 Test the tagger.-----------------------------------------------------------
    testAndPrintAcurracies(tagger, X_test, y_test)

    #########################################################
    #                   PART 2
    #train the generic (base) trigram language model; save it
    #########################################################

    trigramLM = trigramModel(taggedSents)
    ###########################################
    #                   PART 3
    #pull 1000 tweets that contain named_entity
    ###########################################
    open('tweets.txt', 'w').close()
    print("\n[Fetching 1000 tweets on topic " + named_entity + "]\n")
    pullTweets(named_entity, twitter_access_token, twitter_access_token_secret)
    print("[Clearing out old tweets from 'tweets.txt'")
    #########################################################################
    #                       PART 4
    #Limit to tweets with the named entity being classified as a named entity
    #########################################################################
    print("[Updating language model with live tweets]\n")
    oldTrigramLM = copy.deepcopy(trigramLM)

    newTrigramLM = updateLanguageModel(taggedSents)

    ##############################################################
    #                          PART 5
    #generate five different phrases that follow the named_entity.
    ##############################################################
    print("\n[Now attempting to create phrases:]\n")
    i=0
    while i < 5:
        print(generatePhrase(named_entity, newTrigramLM))
        i += 1
    return

def generatePhrase(named_entity, model):
    phrase = ""

    new_word_prob = model.bigram[(named_entity,)]

    #get a weighted random word from probabilities 
    probs = list(new_word_prob.items())
    weights = []
    possibleWords = []
    for wor in probs:
        weights.append(wor[-1])
        possibleWords.append(wor[0])

    new_word = choice(possibleWords, p=weights)
    generated_bigram = (named_entity, new_word)
    
    phrase = generated_bigram[0] + " " + generated_bigram[1]
    i=0 #current index into generated phrase
    while i < 5 and new_word != 'END':
        #get probability distribution for next word based on last two words
        probs = model.trigram[generated_bigram].items()

        #pick a next word from the probability distribution (weighted random)
        weights = []
        possibleWords = []
        for wor in probs:
            weights.append(wor[-1])
            possibleWords.append(wor[0])

        new_word = choice(possibleWords, p=weights)
        phrase += " " + new_word
        #update last_bigram 
        generated_bigram = (generated_bigram[1], new_word)
        i+=1
    phrase += '.'

    return phrase


def pullTweets(entity_name, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
   
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    
    myStream.filter(languages=["en"], track=[entity_name])
    return
def updateLanguageModel(taggedSents):
    #get the saved tweets from file
    tweets = []
    c = 0
    with open('tweets.txt') as file:
        for line in file:
            linestr = json.loads(line)
            tweets.append(linestr[1:-1])
            c += 1
    newTokens = []
    
    print("[Done: 'tweets.txt' has been updated]")
    #get a list of the new tokens and add wordsList words
    for tweetStr in tweets:
        newTokens.append(tokenize(tweetStr))

    #add them to LM
    #get old list of words
    wordsList = []
    for sent in taggedSents:
        if sent: 
            words, tags = zip(*sent)
            for i in range(len(words)):
                wordsList.append(words[i])
    for t in newTokens:
        for word in t:
            wordsList.append(word)

    #create updated model from all tokens
    model = trigramModelFromWordList(wordsList)

    return model

def trigramModel(taggedSents):
    wordsList = []
    for sent in taggedSents:
        if sent: 
            words, tags = zip(*sent)
            for i in range(len(words)):
                wordsList.append(words[i])

    trigramModel = trigramModelFromWordList(wordsList)
    return trigramModel


def trigramModelFromWordList(wordsList):

    countListBigram = [tuple(wordsList[i:i+2]) for i in range(len(wordsList)-1)]    
    countListTrigram = [tuple(wordsList[i:i+3]) for i in range(len(wordsList)-2)]

    #count of unigram, bigrams, and trigrams
    listToDictUnigram = {unigram: wordsList.count(unigram) for unigram in set(wordsList)}
    listToDictBigram = {bigram: countListBigram.count(bigram) for bigram in set(countListBigram)}
    listToDictTrigram = {trigram: countListTrigram.count(trigram) for trigram in set(countListTrigram)}
    
    #do the trigram model probabilities
    ngramCounts = listToDictTrigram
    trigramModelProbs = dict()# stores p(Xi|Xi-1), [x--k...x-1][xi]

    for ngram, count in ngramCounts.items():
        p = count / listToDictBigram[ngram[0:-1]]
        try: 
            trigramModelProbs[ngram[0:-1]][ngram[-1]] = p #indexed by [x--k...x-1][xi]
        except KeyError:
            trigramModelProbs[ngram[0:-1]] = {ngram[-1]: p}
    
    #do the bigram model probabilities
    ngramCounts = listToDictBigram
    bigramModelProbs = dict()# stores p(Xi|Xi-1), [x--k...x-1][xi]
    for ngram, count in ngramCounts.items():
        p = count / listToDictUnigram[ngram[0]]
        try: 
            bigramModelProbs[ngram[0:-1]][ngram[-1]] = p #indexed by [x--k...x-1][xi]
        except KeyError:
            bigramModelProbs[ngram[0:-1]] = {ngram[-1]: p}
    
    model = NgramModel()
    model.trigram = trigramModelProbs
    model.bigram = bigramModelProbs
    return model

def getFeaturesForNER(tokens, targetI, wordToIndex, wordToIndexMultiple, nounList):
    np.set_printoptions(threshold=sys.maxsize)

    #is the word capitalized
    wordCap = np.array([1 if tokens[targetI][0].isupper() else 0])
    oovIndex = len(wordToIndex)

    #first letter of the target word
    letterVec = np.zeros(257)
    val = ord(tokens[targetI][0])
    if val < 256:
        letterVec[val] = 1
    else:
        letterVec[256] = 1
        
    #length of the word:
    length = np.array([len(tokens[targetI])])
    
    #previousWord:
    prevVec = np.zeros(len(wordToIndex)+1)#+1 for OOV
    if targetI > 0:
        try:
            prevVec[wordToIndex[tokens[targetI - 1]]] = 1
        except KeyError:
            prevVec[oovIndex] = 1
            pass#no features added

    #targetWord:
    targetVec = np.zeros(len(wordToIndex)+1)
    try:
        targetVec[wordToIndex[tokens[targetI]]] = 1
    except KeyError:
        targetVec[oovIndex] = 1
        #print("unable to find wordIndex for '", tokens[targetI], "' skipping")
        pass
    
    #nextWord
    nextVec = np.zeros(len(wordToIndex)+1)
    if targetI+1 < len(tokens) :
        try:
            nextVec[wordToIndex[tokens[targetI + 1]]] = 1
        except KeyError:
            nextVec[oovIndex] = 1
            pass
    ############################################
    #new features added for generating sentences
    ############################################
     
    #feature for if word is first word of sentence
    firstWordSentence = np.array([0 if targetI>0 else 1])

    #out of vocab check
    outVocab = [0]
    if tokens[targetI] not in wordToIndexMultiple:
        outVocab = [1]

    #common noun PREV word 
    if tokens[targetI-1] in nounList:
        commonNounPrev = [1]
    else:
        commonNounPrev = [0]
    #common noun TARGET word 
    if tokens[targetI] in nounList:
        commonNounTarg = [1]
    else:
        commonNounTarg = [0]
    #common noun NEXT word 
    
    if targetI == len(tokens) and tokens[targetI+1] in nounList:
        commonNounNext = [1]
    else:
        commonNounNext = [0]

    featureVector = np.concatenate((wordCap, letterVec, length, prevVec,targetVec, nextVec,\
                                   firstWordSentence, outVocab,\
                                    commonNounPrev, commonNounTarg, commonNounNext))

    return featureVector

def trainTagger(features, tags):
    #inputs: features: feature vectors (i.e. X)
    #        tags: tags that correspond to each feature vector (i.e. y)
    #output: model -- a trained (i.e. fit) sklearn.lienear_model.LogisticRegression model
    #print(features[:3], tags[:3])
    
    #train different models and pick the best according to a development set:
    Cs = [.001, .01, .1, 1, 10, 100, 1000, 10000]
    penalties = ['l1', 'l2']
    from sklearn.model_selection import train_test_split
    X_train, X_dev, y_train, y_dev = train_test_split(features, tags,
                                                    test_size=0.20,
                                                    random_state=42)
    bestAcc = 0.0
    bestModel = None
    for pen in penalties: #l1 or l2
        for c in Cs: #c values:
            model = LogisticRegression(random_state=42, penalty=pen, multi_class='auto',\
                                       solver='liblinear', C = c)
            model.fit(X_train, y_train)
            modelAcc = metrics.accuracy_score(y_dev, model.predict(X_dev))
            if modelAcc > bestAcc:
                bestModel = model
                bestAcc = modelAcc
    
    print("Chosen Best Model: \n", bestModel, "\nACC: %.3f"%bestAcc)
    
    return bestModel



def testAndPrintAcurracies(tagger, features, true_tags):
    #inputs: tagger: an sklearn LogisticRegression object to perform tagging
    #        features: feature vectors (i.e. X)
    #        true_tags: tags that correspond to each feature vector (i.e. y)     
    
    pred_tags = tagger.predict(features)
    print("\nModel Accuracy: %.3f" % metrics.accuracy_score(true_tags, pred_tags))
    #most Frequent Tag: 
    mfTags = [Counter(true_tags).most_common(1)[0][0]]*len(true_tags) 
    print("MostFreqTag Accuracy: %.3f" % metrics.accuracy_score(true_tags, mfTags))
    
    return

def getConllTags(filename):
    #input: filename for a conll style parts of speech tagged file
    #output: a list of list of tuples [sent]. representing [[[word1, tag], [word2, tag2]]
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f: 
            wordtag=wordtag.strip()
            if wordtag:#still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word,tag))
            else:#new sentence
                wordTagsPerSent.append([])
                sentNum+=1
    return wordTagsPerSent 


def getConllList(filename):
    #input: filename for a conll style list of words
    #output: a list of words
    words = []
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for word in f: 
            word=word.strip()
            if word:#still reading current line
                words.append(word)
    return words 

if __name__ == "__main__":
    #run the named entity generetive summary
    #named_entity = the topic to generate a sentence from (one word)
    named_entity="California"

    
    #call the application method
    NamedEntityGenerativeSummary(named_entity, 
                        twitter_access_token, twitter_access_token_secret)



