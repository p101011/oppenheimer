from collections import defaultdict
import csv
import random


class MarkovGenerator:
    def __init__(self, order=4):
        self.order = order
        self.freq_dict = defaultdict(list)
        self.invalid_chars = ['?', '/', '\\']

    def load_csv(self, csvfile):
        """Takes a csv and adds words to the markov chain table"""
        with open(csvfile) as open_file:
            readcsv = csv.reader(open_file, delimiter=',')
            for row in readcsv:
                for word in row:
                    self.load_word(word)

    def load_word_list(self, word_list):
        """Takes a list of words seperated by spaces and generates a markov chain table"""
        for word in word_list:
            self.load_word(word)

    def load_word(self, s):
        for char in range(len(s) - self.order):
            if char not in self.invalid_chars:
                self.freq_dict[s[char:char + self.order]].append(s[char + self.order])
    
    def generate(self, start=None, max_length=15):
        """This function takes a Markov Dictionary and optionally a starting char and
        word length and returns a randomly generated word."""
        if start is None:
            s = random.choice(list(self.freq_dict))
            while not s.isalpha():
                s = random.choice(list(self.freq_dict))
        else:
            s = start
        try:
            while len(s) < max_length:
                next_char = random.choice(self.freq_dict[s[-self.order:]])
                if next_char != ' ' and next_char.isalpha():
                    s += next_char
                else:
                    break
        except IndexError:
            pass
        return s

    def __repr__(self):
        return str(self.freq_dict)
