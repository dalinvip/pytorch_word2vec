import numpy
from collections import deque

from zmq.backend.cython import context

from huffman import HuffmanTree
numpy.random.seed(12345)


class InputData:
    """Store data for word2vec, such as word map, huffman tree, sampling table and so on.

    Attributes:
        word_frequency: Count of each word, used for filtering low-frequency words and sampling table
        word2id: Map from word to word id, without low-frequency words.
        id2word: Map from word id to word, without low-frequency words.
        sentence_count: Sentence count in files.
        word_count: Word count in files, without low-frequency words.
    """

    def __init__(self, file_name, min_count):
        self.get_words(file_name, min_count)
        print(" ")
        self.cbow_count = []
        self.word_pair_catch = deque()
        self.cbow_word_pair_catch = deque()
        self.init_sample_table()
        tree = HuffmanTree(self.word_frequency)
        print("tree ", tree)
        self.huffman_positive, self.huffman_negative = tree.get_huffman_code_and_path()
        print('Word Count: %d' % len(self.word2id))
        print('Sentence Length: %d' % (self.sentence_length))

    def get_words(self, file_name, min_count):
        self.input_file_name = file_name
        # self.input_file = open(self.input_file_name)
        self.input_file = open(self.input_file_name, encoding="UTF-8")
        self.sentence_length = 0
        self.sentence_count = 0
        word_frequency = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    word_frequency[w] += 1
                except:
                    word_frequency[w] = 1
        self.word2id = dict()
        self.id2word = dict()
        wid = 0
        self.word_frequency = dict()
        for w, c in word_frequency.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        self.word_count = len(self.word2id)

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        # pow_frequency = numpy.array(self.word_frequency.values())**0.75
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    # @profile
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            for _ in range(10000):
                sentence = self.input_file.readline()
                if sentence is None or sentence == '':
                    self.input_file = open(self.input_file_name, encoding="utf-8")
                    sentence = self.input_file.readline()
                word_ids = []
                for word in sentence.strip().split(' '):
                    try:
                        word_ids.append(self.word2id[word])
                    except:
                        continue
                for i, u in enumerate(word_ids):
                    for j, v in enumerate(word_ids[max(i - window_size, 0):i + window_size]):
                        assert u < self.word_count
                        assert v < self.word_count
                        if i == j:
                            continue
                        self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs

    def get_cbow_batch_all_pairs(self, batch_size, context_size):
        while len(self.cbow_word_pair_catch) < batch_size:
            for _ in range(10000):
                self.input_file = open(self.input_file_name, encoding="utf-8")
                sentence = self.input_file.readline()
                if sentence is None or sentence == '':
                    continue
                    # self.input_file = open(self.input_file_name, encoding="utf-8")
                    # sentence = self.input_file.readline()
                # if sentence is not None or sentence != "":
                word_ids = []
                for word in sentence.strip().split(' '):
                    try:
                        word_ids.append(self.word2id[word])
                    except:
                        continue
                # for i, u in enumerate(word_ids):
                #     con = []
                #     for j, v in enumerate(word_ids[max(i - window_size, 0):i + window_size]):
                #        assert u < self.word_count
                #        assert v < self.word_count
                #        if i == j:
                #            continue
                #        elif j >= max(0, i - window_size + 1) and j <= min(len(word_ids), i + window_size - 1):
                #            con.append(v)
                #     if len(con) == 0:
                #         continue
                #     self.cbow_word_pair_catch.append((con, u))

                # for i in range(2, len(word_ids) - 2):
                #     bow = ([word_ids[i - 2], word_ids[i - 1], word_ids[i + 1], word_ids[i + 2]], word_ids[i])
                #     self.cbow_word_pair_catch.append(bow)

                for i, u in enumerate(word_ids):
                    contentw = []
                    for j, v in enumerate(word_ids):
                        assert u < self.word_count
                        assert v < self.word_count
                        if i == j:
                            continue
                        # elif j >= max(0, i - self.args.window_size + 1) and j <= min(len(word_ids), i + self.args.window_size-1):
                        elif j >= max(0, i - context_size + 1) and j <= min(len(word_ids), i + context_size-1):
                            contentw.append(v)
                    if len(contentw) == 0:
                        continue
                    self.cbow_word_pair_catch.append((contentw, u))

                # for i in range(context_size, len(word_ids) - context_size):
                #     context = []
                #     for j in range(context_size, 0, -1):
                #         context.append(word_ids[i - j])
                #     for j in range(1, context_size + 1):
                #         context.append(word_ids[i + j])
                #     bow = (context, word_ids[i])
                #     self.cbow_word_pair_catch.append(bow)
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.cbow_word_pair_catch.popleft())
        return batch_pairs


    def get_cbow_batch_pairs(self, batch_size, window_size):
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.cbow_word_pair_catch.popleft())
        return batch_pairs

    # @profile
    def get_pairs_by_neg_sampling(self, pos_word_pair, count):
        neg_word_pair = []
        a = len(self.word2id) - 1
        for pair in pos_word_pair:
            i = 0
            neg_v = numpy.random.choice(self.sample_table, size=count)
            neg_word_pair += zip([pair[0]] * count, neg_v)
        return pos_word_pair, neg_word_pair

    def get_cbow_pairs_by_neg_sampling(self, pos_word_pair, count):
        neg_word_pair = []
        # print("get_cbow_pairs_by_neg_sampling", pos_word_pair)
        for pair in pos_word_pair:
            neg_v = numpy.random.choice(self.sample_table, size=count)
            neg_word_pair += zip([pair[0]] * count, neg_v)
        return pos_word_pair, neg_word_pair

    def get_pairs_by_huffman(self, pos_word_pair):
        neg_word_pair = []
        a = len(self.word2id) - 1
        for i in range(len(pos_word_pair)):
            pair = pos_word_pair[i]
            pos_word_pair += zip([pair[0]] *
                                 len(self.huffman_positive[pair[1]]),
                                 self.huffman_positive[pair[1]])
            neg_word_pair += zip([pair[0]] *
                                 len(self.huffman_negative[pair[1]]),
                                 self.huffman_negative[pair[1]])

        return pos_word_pair, neg_word_pair

    def get_cbow_pairs_by_huffman(self, pos_word_pair):
        neg_word_pair = []
        a = len(self.word2id) - 1
        for i in range(len(pos_word_pair)):
            pair = pos_word_pair[i]
            pos_word_pair += zip([pair[0]] *
                                 len(self.huffman_positive[pair[1]]),
                                 self.huffman_positive[pair[1]])
            neg_word_pair += zip([pair[0]] *
                                 len(self.huffman_negative[pair[1]]),
                                 self.huffman_negative[pair[1]])
        # print("hufman pos", pos_word_pair)
        # print("hufman nef", neg_word_pair)
        return pos_word_pair, neg_word_pair


    def evaluate_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size


def test():
    a = InputData('./zhihu.txt')


if __name__ == '__main__':
    test()
