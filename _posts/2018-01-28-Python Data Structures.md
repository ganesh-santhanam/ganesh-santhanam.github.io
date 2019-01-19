---
title: "Machine Learning Project: Perceptron"
date: 2018-01-28
tags: [machine learning, data science, neural network]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Machine Learning, Perceptron, Data Science"
mathjax: "true"
---

# H1 Heading

## H2 Heading

### H3 Heading

Here's some basic text.

And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third



# Big O Notation

Big O notation is used in Computer Science to describe the performance or complexity of an algorithm. Big O specifically describes the worst-case scenario, and can be used to describe the execution time required or the space used

<p align="center">
<img src="https://imgur.com/PNQgBup.jpg">

</p>

<center>
 Common functions used in algorithm analysis
</center>


<p align="center">
<img src="https://imgur.com/jDFNAsN.jpg">

</p>

<center>
 Growth rates of Common functions used in algorithm analysis
</center>

Source Goodrich and Tamassia


# Recursion

An alternative to iteration using for or while loops is recursion. Recursion is a technique by which a function makes one or more calls to itself during execution

A Recursive Implementation of the Factorial Function

Python code block:
```python
def factorial(n):
   """Function to return the factorial
   of a number using recursion"""
   if n == 0:
       return n
   else:
       return n*factorial(n-1)
```
The time efficiency is O(n), as there are n + 1 activations, each of which accounts for O(1) operations.

One concern with using recursion over iteration is the limit of recursion depth for the function calls ( typical default value is 1000 ). If this limit is reached, the Python interpreter raises a RuntimeError with a message, maximum recursion depth exceeded. Also in an interpreted language such as Python recursion may run slower than iterative alternatives.

# Dynamic Arrays

When creating a low-level array in a computer system, the precise size of that array must be explicitly declared . However Python's list class allows us to add elements to the list using an abstraction called Dynamic Array.

 If an element is appended to a list at a time when the underlying array is full, we perform the following steps:

1.Allocate a new array B with larger capacity(Usually double the size of the existing array).
2.Set B[i] = A[i], for i = 0 to n – 1
3.Set A = B, that is use B as the array
4.Insert the new element in the new array


<p align="center">
<img src="https://imgur.com/YonzVJc.jpg">

</p>

<center>
 Dynamic Array
</center>

Python code block:
```python
import ctypes  # provides low-level arrays

class DynamicArray(object):    

    def __init__(self):
        self.n = 0 # Count actual elements (Default is 0)
        self.capacity = 1 # Default Capacity
        self.A = self.make_array(self.capacity)

    def __len__(self):
        """
        Return number of elements sorted in array
        """
        return self.n

    def __getitem__(self,k):
        """
        Return element at index k
        """
        if not 0 <= k <self.n:
            return IndexError('K is out of bounds!') # Check it k index is in bounds of array

        return self.A[k] #Retrieve from array at index k

    def append(self, ele):
        """
        Add element to end of the array
        """
        if self.n == self.capacity:
            self._resize(2*self.capacity) #Double capacity if not enough room

        self.A[self.n] = ele #Set self.n index to element
        self.n += 1

    def _resize(self,new_cap):
        """
        Resize internal array to capacity new_cap
        """

        B = self.make_array(new_cap) # New bigger array

        for k in range(self.n): # Reference all existing values
            B[k] = self.A[k]

        self.A = B # Call A the new bigger array
        self.capacity = new_cap # Reset the capacity

    def make_array(self,new_cap):
        """
        Returns a new array with new_cap capacity
        """
        return (new_cap * ctypes.py_object)()

 ```
The amortized running time of each append operation is O(1); hence, the total running time of n append operations is O(n).

# Stack

A stack is a collection of objects that are inserted and removed according to the last-in, first-out (LIFO) principle.

```python
class Stack:

	def __init__(self):
		self.stack = []

	def isEmpty(self):
		return self.stack == []

	def push(self, data):
		self.stack.append(data)

	def pop(self):
		data = self.stack[-1]
		del self.stack[-1]
		return data

	def top(self):
		return self.stack[-1]

	def sizeStack(self):
		return len(self.stack)
```

Time complexity is O(1) time for push and pop (amortized bounds)

# Queues
Queue is a collection of objects that are inserted and removed according to the first-in, first-out (FIFO) principle.

```python
class Queue:

	def __init__(self):
		self.queue = []

	def isEmpty(self):
		return self.queue == []

	def enqueue(self, data):
		self.queue.append(data)

	def dequeue(self):
		data = self.queue[0]
		del self.queue[0]
		return data

	def top(self):
		return self.queue[0]

	def sizeQueue(self):
		return len(self.queue)
  ```

Enqueue and Dequeue, which have amortized bounds of O(1) time

# Double-Ended Queues

Queue-like data structure that supports insertion and deletion at both the front and the back of the queue  is called a double-ended queue, or deque
```Python
class Deque:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def addFront(self, item):
        self.items.append(item)

    def addRear(self, item):
        self.items.insert(0,item)

    def removeFront(self):
        return self.items.pop()

    def removeRear(self):
        return self.items.pop(0)

    def size(self):
        return len(self.items)
  ```

 The deque class has O(1)-time operations at either end, but O(n)-time when operating near the middle of the deque.

# Singly Linked Lists
A singly linked list, in its simplest form, is a collection of nodes that collectively form a linear sequence. Each node stores a reference to an object that is an element of the sequence, as well as a reference to the next node of the list
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/6/6d/Singly-linked-list.svg">

</p>

<center>
 singly linked list
</center>

```Python
class Node(object):

    def __init__(self,value):

        self.value = value
        self.nextnode = None
```

# Doubly Linked Lists
A linked list in which each node keeps a reference to the node before it and a reference to the node after it, is known as a doubly linked list.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Doubly-linked-list.svg">

</p>

<center>
 Double linked list
</center>

```python
class DoublyLinkedListNode(object):

    def __init__(self,value):

        self.value = value
        self.next_node = None
        self.prev_node = None
```

#Link-Based vs. Array-Based Sequences

##Advantages of Array-Based Sequences

* Arrays provide O(1)-time access to any element. In contrast, locating the kth element in a linked list requires O(k) or possibly O(n – k) time
* Array-based representations typically use proportionally less memory than linked structures as memory must be devoted to references that link the nodes.

##Advantages of Link-Based Sequences
* Link-based structures support O(1)-time insertions and deletions at arbitrary positions.


#Binary Search Trees
A binary search tree is a binary tree which  additionally satisfies the binary search property. The key in each node must be greater than or equal to any key stored in the left sub-tree, and less than or equal to any key stored in the right sub-tree

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/d/da/Binary_search_tree.svg">

</p>

<center>
 Binary Search Tree
</center>

```python
or
class Node(object):

	def __init__(self, data):
		self.data = data;
		self.leftChild = None;
		self.rightChild = None;

class BinarySearchTree(object):

	def __init__(self):
		self.root = None;

	def insert(self, data):
		if not self.root:
			self.root = Node(data);
		else:
			self.insertNode(data, self.root);

	def insertNode(self, data, node):

		if data < node.data:
			if node.leftChild:
				self.insertNode(data, node.leftChild);
			else:
				node.leftChild = Node(data);
		else:
			if node.rightChild:
				self.insertNode(data, node.rightChild);
			else:
				node.rightChild = Node(data);

	def removeNode(self, data, node):

		if not node:
			return node;

		if data < node.data:
			node.leftChild = self.removeNode(data, node.leftChild);
		elif data > node.data:
			node.rightChild = self.removeNode(data, node.rightChild);
		else:

			if not node.leftChild and not node.rightChild:
				print("Removing a leaf node...");
				del node;
				return None;

			if not node.leftChild:  
				print("Removing a node with single right child...");
				tempNode = node.rightChild;
				del node;
				return tempNode;
			elif not node.rightChild:
				print("Removing a node with single left child...");
				tempNode = node.leftChild;
				del node;
				return tempNode;

			print("Removing node with two children....");
			tempNode = self.getPredecessor(node.leftChild);   
			node.data = tempNode.data;
			node.leftChild = self.removeNode(tempNode.data, node.leftChild);

		return node;

	def getPredecessor(self, node):

		if node.rightChild:
			return self.getPredeccor(node.rightChild);

		return node;

	def remove(self, data):
		if self.root:
			self.root = self.removeNode(data, self.root);


	def getMinValue(self):
		if self.root:
			return self.getMin(self.root);

	def getMin(self, node):

		if node.leftChild:
			return self.getMin(node.leftChild);

		return node.data;

	def getMaxValue(self):
		if self.root:
			return self.getMax(self.root);

	def getMax(self, node):

		if node.rightChild:
			return self.getMax(node.rightChild);

		return node.data;

	def traverse(self):
		if self.root:
			self.traverseInOrder(self.root);		

	def traverseInOrder(self, node):
		if node.leftChild:
			self.traverseInOrder(node.leftChild);

		print("%s " % node.data);

		if node.rightChild:
			self.traverseInOrder(node.rightChild);			

```

There are many ways to traverse this Tre
Depth First Traversals:
(a) Inorder (Left, Root, Right)
(b) Preorder (Root, Left, Right)
(c) Postorder (Left, Right, Root)
(d) Breadth First or Level Order Traversal (visit every node on a level before going to a lower level)

An in-order traversal of a binary search tree will always result in a sorted list of node items. Pre-order traversal or a post-order traversal do not make sense for BST

#Priority queue

This is a collection of prioritized elements that allows arbitrary element insertion, and allows the removal of the element that has first priority. When an element is added to a priority queue, the user designates its priority by providing an associated key. The element with the minimum key will be the next to be removed from the queue

An Efficient realization of a priority queue can be done using a data structure called a binary heap. This data structure allows us to perform both insertions and removals in logarithmic time. Heap is a tree that satisfies the heap property: if P is a parent node of C, then the key (the value) of P is either greater than or equal to (in a max heap) or less than or equal to (in a min heap) the key of C
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/3/38/Max-Heap.svg">

</p>

<center>
 Binary Max Heap
</center>

```python

class Heap(object):
	HEAP_SIZE = 10

	def __init__(self):
		self.heap = [0]*Heap.HEAP_SIZE;
		self.currentPosition = -1;

	def insert(self, item):
		if self.isFull():
			print("Heap is full..");
			return

		self.currentPosition = self.currentPosition + 1
		self.heap[self.currentPosition] = item
		self.fixUp(self.currentPosition)

	def fixUp(self, index):
		parentIndex = int((index-1)/2)		
		while parentIndex >= 0 and self.heap[parentIndex] < self.heap[index]:
			temp = self.heap[index]
			self.heap[index] = self.heap[parentIndex]
			self.heap[parentIndex] = temp;
			parentIndex = (int)((index-1)/2)

	def heapsort(self):
		for i in range(0,self.currentPosition+1):
			temp = self.heap[0]
			print("%d " % temp)
			self.heap[0] = self.heap[self.currentPosition-i]
			self.heap[self.currentPosition-i] = temp
			self.fixDown(0,self.currentPosition-i-1)

	def fixDown(self, index, upto):
		while index <= upto:

			leftChild = 2*index+1
			rightChild = 2*index+2

			if leftChild < upto:
				childToSwap = None

				if rightChild > upto:
					childToSwap = leftChild
				else:
					if self.heap[leftChild] > self.heap[rightChild]:
						childToSwap = leftChild
					else:
						childToSwap = rightChild

				if self.heap[index] < self.heap[childToSwap]:
					temp = self.heap[index]
					self.heap[index] = self.heap[childToSwap]
					self.heap[childToSwap] = temp
				else:
					break

				index = childToSwap
			else:
				break;							

	def isFull(self):
		if self.currentPosition == Heap.HEAP_SIZE:
			return True
		else:
			return False		
```

We can use the heap to sort which is called Heapsort. It is an in-place algorithm with performance of O (n log n)
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/f/fe/Heap_sort_example.gif">

</p>

<center>
 Heap Sort
</center>


#Hash Tables

Hash table  is a data structure that can map keys to values. A hash table uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found.

Ideally, the hash function will assign each key to a unique bucket, but most hash table designs employ an imperfect hash function, which might cause hash collisions where the hash function generates the same index for more than one key.
Chained hash tables with linked lists are popular to resolve collisions.

Load Factor = $= \frac { n } { k }$ where n is the number of entries occupied in the hash table and k is the number of buckets.

Hash table has a performance of $\Theta \left( 1 + \frac { n } { k } \right)$.
As the load factor grows larger, the hash table becomes slower. Hence we may have to resize the hash table.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/d/d0/Hash_table_5_0_1_1_1_1_1_LL.svg">

</p>

<center>
 Hash Table
</center>

```python
class HashTable(object):

    def __init__(self,size):

        # Set up size and slots and data
        self.size = size
        self.slots = [None] * self.size
        self.data = [None] * self.size

    def put(self,key,data):
        #Note, we'll only use integer keys for ease of use with the Hash Function

        # Get the hash value
        hashvalue = self.hashfunction(key,len(self.slots))

        # If Slot is Empty
        if self.slots[hashvalue] == None:
            self.slots[hashvalue] = key
            self.data[hashvalue] = data

        else:

            # If key already exists, replace old value
            if self.slots[hashvalue] == key:
                self.data[hashvalue] = data  

            # Otherwise, find the next available slot
            else:

                nextslot = self.rehash(hashvalue,len(self.slots))

                # Get to the next slot
                while self.slots[nextslot] != None and self.slots[nextslot] != key:
                    nextslot = self.rehash(nextslot,len(self.slots))

                # Set new key, if NONE
                if self.slots[nextslot] == None:
                    self.slots[nextslot]=key
                    self.data[nextslot]=data

                # Otherwise replace old value
                else:
                    self.data[nextslot] = data

    def hashfunction(self,key,size):
        # Remainder Method
        return key%size

    def rehash(self,oldhash,size):
        # For finding next possible positions
        return (oldhash+1)%size


    def get(self,key):

        # Getting items given a key

        # Set up variables for our search
        startslot = self.hashfunction(key,len(self.slots))
        data = None
        stop = False
        found = False
        position = startslot

        # Until we discern that its not empty or found (and haven't stopped yet)
        while self.slots[position] != None and not found and not stop:

            if self.slots[position] == key:
                found = True
                data = self.data[position]

            else:
                position=self.rehash(position,len(self.slots))
                if position == startslot:

                    stop = True
        return data

    # Special Methods for use with Python indexing
    def __getitem__(self,key):
        return self.get(key)

    def __setitem__(self,key,data):
        self.put(key,data)

```

#Balanced Search Trees
A standard binary search tree supports O(logn) expected running times for the basic operations. However, we may only claim O(n) worst-case time, because some sequences of operations may lead to an unbalanced tree with height proportional to n. A self-balancing binary search tree is one that automatically keeps its height (maximal number of levels below the root) small in the face of arbitrary item insertions and deletions.

A commonly used implementation of this is Red Black tree
<p align="center">
<img src="https://zippy.gfycat.com/SkinnyUnluckyBlackrussianterrier.gif">

</p>

<center>
 Red Black Tree
</center>

```python
from .binary_search_tree import TreeMap

class RedBlackTreeMap(TreeMap):
  """Sorted map implementation using a red-black tree."""

  #-------------------------- nested _Node class --------------------------
  class _Node(TreeMap._Node):
    """Node class for red-black tree maintains bit that denotes color."""
    __slots__ = '_red'     # add additional data member to the Node class

    def __init__(self, element, parent=None, left=None, right=None):
      super().__init__(element, parent, left, right)
      self._red = True     # new node red by default

  #------------------------- positional-based utility methods -------------------------
  # we consider a nonexistent child to be trivially black
  def _set_red(self, p): p._node._red = True
  def _set_black(self, p): p._node._red = False
  def _set_color(self, p, make_red): p._node._red = make_red
  def _is_red(self, p): return p is not None and p._node._red
  def _is_red_leaf(self, p): return self._is_red(p) and self.is_leaf(p)

  def _get_red_child(self, p):
    """Return a red child of p (or None if no such child)."""
    for child in (self.left(p), self.right(p)):
      if self._is_red(child):
        return child
    return None

  #------------------------- support for insertions -------------------------
  def _rebalance_insert(self, p):
    self._resolve_red(p)                         # new node is always red

  def _resolve_red(self, p):
    if self.is_root(p):
      self._set_black(p)                         # make root black
    else:
      parent = self.parent(p)
      if self._is_red(parent):                   # double red problem
        uncle = self.sibling(parent)
        if not self._is_red(uncle):              # Case 1: misshapen 4-node
          middle = self._restructure(p)          # do trinode restructuring
          self._set_black(middle)                # and then fix colors
          self._set_red(self.left(middle))
          self._set_red(self.right(middle))
        else:                                    # Case 2: overfull 5-node
          grand = self.parent(parent)            
          self._set_red(grand)                   # grandparent becomes red
          self._set_black(self.left(grand))      # its children become black
          self._set_black(self.right(grand))
          self._resolve_red(grand)               # recur at red grandparent

  #------------------------- support for deletions -------------------------
  def _rebalance_delete(self, p):
    if len(self) == 1:                                     
      self._set_black(self.root())  # special case: ensure that root is black
    elif p is not None:
      n = self.num_children(p)
      if n == 1:                    # deficit exists unless child is a red leaf
        c = next(self.children(p))
        if not self._is_red_leaf(c):
          self._fix_deficit(p, c)
      elif n == 2:                  # removed black node with red child
        if self._is_red_leaf(self.left(p)):
          self._set_black(self.left(p))
        else:
          self._set_black(self.right(p))

  def _fix_deficit(self, z, y):
    """Resolve black deficit at z, where y is the root of z's heavier subtree."""
    if not self._is_red(y): # y is black; will apply Case 1 or 2
      x = self._get_red_child(y)
      if x is not None: # Case 1: y is black and has red child x; do "transfer"
        old_color = self._is_red(z)
        middle = self._restructure(x)
        self._set_color(middle, old_color)   # middle gets old color of z
        self._set_black(self.left(middle))   # children become black
        self._set_black(self.right(middle))
      else: # Case 2: y is black, but no red children; recolor as "fusion"
        self._set_red(y)
        if self._is_red(z):
          self._set_black(z)                 # this resolves the problem
        elif not self.is_root(z):
          self._fix_deficit(self.parent(z), self.sibling(z)) # recur upward
    else: # Case 3: y is red; rotate misaligned 3-node and repeat
      self._rotate(y)
      self._set_black(y)
      self._set_red(z)
      if z == self.right(y):
        self._fix_deficit(z, self.left(z))
      else:
        self._fix_deficit(z, self.right(z))

```


From Goodrich and Tamassia book

<p align="center">
<img src="https://he-s3.s3.amazonaws.com/media/uploads/c14cb1f.JPG"">

</p>

<center>
 Comparison of common data structures
</center>
