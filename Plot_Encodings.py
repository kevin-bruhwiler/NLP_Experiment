import matplotlib.pyplot as plt
import numpy as np
import random
import pickle


def run():
    with open('data/embeddings.npy', 'rb') as f:
        embeddings = np.load(f)
    with open('data/word_to_ix.pkl', 'rb') as f:
        word_to_ix = pickle.load(f)
    with open('data/words.pkl', 'rb') as f:
        words = pickle.load(f).split()

    s = random.sample(words, 250)
    n = [embeddings[word_to_ix[word]] for word in s]
    x = [num[0] for num in n]
    y = [num[1] for num in n]

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(s):
        ax.annotate(txt, (x[i],y[i]))

    plt.show()

if __name__=="__main__":
    print("Beginning program....")
    run()