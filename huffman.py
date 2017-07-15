# import over_heap as heapq
# import heap_file as heapq
# import _heapq as heapq
import heapq as heapq


from anaconda_navigator.utils.py3compat import cmp


class Node:
    def __init__(self, wid, frequency):
        self.wid = wid
        self.frequency = frequency
        self.father = None
        self.is_left_child = None
        self.left_child = None
        self.right_child = None
        self.code = []
        self.path = []

class Heap:
    def __init__(self):
        print("Heap")

    def heappush(heap, item):
        """Push item onto heap, maintaining the heap invariant."""
        heap.append(item)
        Heap.siftdown(heap, 0, len(heap) - 1)

    def siftdown(heap, startpos, pos):
        newitem = heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = heap[parentpos]
            if newitem[0] == parent[0]:
                if(str(newitem) > str(parent)):
                   heap[pos] = parent
                   pos = parentpos
                   continue
            else:
                if newitem < parent:
                    heap[pos] = parent
                    pos = parentpos
            break
        heap[pos] = newitem

    def siftdown2(heap, startpos, pos):
        newitem = heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = heap[parentpos]
            if str(newitem) >= str(parent):
                heap[pos] = parent
                pos = parentpos
                continue
            break
        heap[pos] = newitem


    def heappop(heap):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
        if heap:
            returnitem = heap[0]
            heap[0] = lastelt
            Heap.siftup(heap, 0)
            return returnitem
        return lastelt

    def siftup(heap, pos):
        endpos = len(heap)
        startpos = pos
        newitem = heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not str(heap[childpos]) <= str(heap[rightpos]):
                childpos = rightpos
            # Move the smaller child up.
            heap[pos] = heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        heap[pos] = newitem
        Heap.siftdown(heap, startpos, pos)




class HuffmanTree:
    def __init__(self, word_frequency):
        self.word_count = len(word_frequency)
        self.huffman = []
        unmerged_node = []
        word_frequency_list = []
        for index, value in word_frequency.items():
            word_frequency_list.append(value)
        # print(word_frequency_list)
        for wid, c in word_frequency.items():
        # for wid, c in enumerate(word_frequency_list):
            node = Node(wid, c)
            heapq.heappush(unmerged_node, (c, wid, node))
            self.huffman.append(node)
        next_id = len(self.huffman)
        while len(unmerged_node) > 1:
            _, _, node1 = heapq.heappop(unmerged_node)
            _, _, node2 = heapq.heappop(unmerged_node)
            new_node = Node(next_id, node1.frequency + node2.frequency)
            node1.father = new_node.wid
            node2.father = new_node.wid
            new_node.left_child = node1.wid
            node1.is_left_child = True
            new_node.right_child = node2.wid
            node2.is_left_child = False
            self.huffman.append(new_node)
            heapq.heappush(unmerged_node, (new_node.frequency, new_node.wid, new_node))
            next_id = len(self.huffman)


        self.get_huffman_code(unmerged_node[0][2].left_child)
        self.get_huffman_code(unmerged_node[0][2].right_child)


# class HuffmanTree:
#     def __init__(self, word_frequency):
#         self.word_count = len(word_frequency)
#         self.huffman = []
#         unmerged_node = []
#         word_frequency_list = []
#         for index, value in word_frequency.items():
#             word_frequency_list.append(value)
#         print(word_frequency_list)
#         flag = True
#         for wid, c in word_frequency.items():
#         # for wid, c in enumerate(word_frequency_list):
#             node = Node(wid, c)
#             Heap.heappush(unmerged_node, (c, node))
#             self.huffman.append(node)
#         next_id = len(self.huffman)
#         while len(unmerged_node) > 1:
#             _, node1 = Heap.heappop(unmerged_node)
#             _, node2 = Heap.heappop(unmerged_node)
#             new_node = Node(next_id, node1.frequency + node2.frequency)
#             node1.father = new_node.wid
#             node2.father = new_node.wid
#             new_node.left_child = node1.wid
#             node1.is_left_child = True
#             new_node.right_child = node2.wid
#             node2.is_left_child = False
#             self.huffman.append(new_node)
#             Heap.heappush(unmerged_node, (new_node.frequency, new_node))
#             next_id = len(self.huffman)
#
#
#         self.get_huffman_code(unmerged_node[0][1].left_child)
#         self.get_huffman_code(unmerged_node[0][1].right_child)


    def get_huffman_code(self, wid):
        # print("huffman code", wid)
        if self.huffman[wid].is_left_child:
            code = [0]
        else:
            code = [1]

        self.huffman[wid].code = self.huffman[self.huffman[wid].father].code + code
        self.huffman[wid].path = self.huffman[self.huffman[wid].father].path + [self.huffman[wid].father]

        if self.huffman[wid].left_child is not None:
            self.get_huffman_code(self.huffman[wid].left_child)
        if self.huffman[wid].right_child is not None:
            self.get_huffman_code(self.huffman[wid].right_child)

    def get_huffman_code_and_path(self):
        positive = []
        negative = []
        for wid in range(self.word_count):
            pos = []
            neg = []
            for i, c in enumerate(self.huffman[wid].code):
                if c == 0:
                    pos.append(self.huffman[wid].path[i])
                else:
                    neg.append(self.huffman[wid].path[i])
            positive.append(pos)
            negative.append(neg)
        return positive, negative


if __name__ == '__main__':
    word_frequency = {0: 4, 1: 6, 2: 3, 3: 2, 4: 2}
    # word_frequency = {0: 4, 1: 6, 2: 3, 3: 2, 4: 8}
    tree = HuffmanTree(word_frequency)
    huffman_code, huffman_path = tree.get_huffman_code_and_path()
    print(huffman_code)
    print(huffman_path)
