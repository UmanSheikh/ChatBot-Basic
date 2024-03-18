import json # Dataset File
import pickle # Saving Model
import numpy as np # Preprocessing Data
from keras.models import Sequential # Importing Sequential Models
from keras.layers import Dense, Activation, Dropout # Importing Layers
from keras.optimizers import SGD # Importing Optimizer
import random # For shuffling the features and make numpy array
import nltk # For Natural Language Processing
from nltk.stem import WordNetLemmatizer # stemming, lemmatization seeks to distill words to their foundational forms


lemmatizer: WordNetLemmatizer = WordNetLemmatizer() # Creating Lemmatizer Object


intents_file: str = open('intents.json').read() # Reading Dataset from the json file
intents: dict = json.loads(intents_file) # Loading json data and converting it to Python dictionary

words: list[str] = [] # Creating Empty list for words
classes: list[str] = [] # Creating Empty list for classes
documents: list[str] = [] # Creating Empty List for documents
ignore_letters: list[str] = ['!', '?', ',', '.'] # List for Ignored letters


for intent in intents['intents']: # First key in dictonary
    for pattern in intent['patterns']: # Finding Patterns
        word: list[str] = nltk.word_tokenize(pattern) # tokenize each word
        words.extend(word) # Adding to Words list
        documents.append((word, intent['tag']))
        if intent['tag'] not in classes: # add to our classes list
            classes.append(intent['tag'])
print(documents) # Printing for results

""" 
We can convert words into the lemma form so that we can reduce all the canonical words. For example, the words play, playing, plays, played, etc. will all be replaced with play. This way, we can reduce the number of total words in our vocabulary. 
"""
# lemmaztize and lower each word and remove duplicates
words: list[str] = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words: list[str] = sorted(list(set(words)))
classes: list[str] = sorted(list(set(classes)))
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


training: list[str] = [] # create the training data
output_empty: list[str] = [0] * len(classes) # create empty array for the output
for doc in documents:
    bag: list[str] = [] # initializing bag of words
    word_patterns: list[str] = doc[0] # list of tokenized words for the pattern
    word_patterns: list[str] = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]  # lemmatize each word - create base word, in attempt to represent related words
    for word in words:  # create the bag of words array with 1, if word is found in current pattern
        bag.append(1) if word in word_patterns else bag.append(0)
        output_row:list[str] = list(output_empty) # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

random.shuffle(training) # shuffle the features and make numpy array
training: list[int] = np.array(training) 
train_x: list[int] = list(training[:,0]) # create training and testing lists. X - patterns, Y - intents
train_y: list[int] = list(training[:,1])
print("Training data is created")

model: Sequential = Sequential() # Creating object of Sequentail Model
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
sgd: SGD = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # Creating Object of SGD with optimizing values
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # Compiling Model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) # Fiting epocs and traning data
model.save('chatbot_model.h5', hist) # Saving Model in chatbot_model.h5
print("model is created")