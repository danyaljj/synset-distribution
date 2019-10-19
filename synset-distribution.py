from nltk.corpus import wordnet
from nltk.corpus import words
import numpy as np
import matplotlib.pyplot as plt

def main():
  synset_size_list = []
  english_words = words.words()
  for word in english_words:
    synsets = wordnet.synsets(word)
    if len(synsets) > 0:
      synset_size_list.append(len(synsets))

  average = sum(synset_size_list)/len(synset_size_list)
  print(f" * count: {len(synset_size_list)}")
  print(f" * average: {average}")

  variance = np.var(synset_size_list)
  print(f" * variance: {variance}")
  print(f" * std: {np.sqrt(variance)}")

  binwidth = 1
  plt.hist(synset_size_list, bins=np.arange(min(synset_size_list), max(synset_size_list) + binwidth, binwidth))
  plt.ylabel('Count')
  plt.show()

if __name__ == '__main__': 
  main()
