---
title: "Advanced Python"
mathjax: "true"
---


# Iterators

Iterators are objects that allow iteration over a collection. Such collections need not be of objects that already exist in memory, and because of this, they need not necessarily be finite. An iterable is defined as an object that has an __iter__ method, which is required to return an iterator object. An iterator in turn is an object that has the two methods __iter__ and  __next__  with the former returning an iterator object and the latter returning the next element of the iteration.

This iterator will simply start counting at 0 and go up indefinitely.  

```python
class count_iterator(object):
    n = 0

    def __iter__(self):
        return self

    def next(self):
        y = self.n
        self.n += 1
        return y
```        
```
>>> counter = count_iterator()
>>> next(counter)
0
>>> next(counter)
1
>>> next(counter)
2
>>> next(counter)
3
```

# Generators  

Generators are iterators that are defined using a simpler function notation. More specifically, a generator is a function that uses the yield expression somewhere in it. Generators can not return values, and instead yield results when they are ready.  

```python
def count_generator():
   n = 0
   while True:
     yield n
     n += 1
Now let's see it in action:
```
```
>>> counter = count_generator()
>>> counter
<generator object count_generator at 0x106bf1aa0>
>>> next(counter)
0
>>> next(counter)
1
```
# Generator Expressions  

Generator expressions let you define generators using an even simpler, inline, notation. This notation is very similar to Python's list comprehension notation. For example, the following provides a generator that iterates over all perfect squares. Note how the results of generator expressions are an objects of type generator, and as such they implement both next and __iter__ methods.

```python
>>> g = (x ** 2 for x in itertools.count(1))
>>> g
<generator object <genexpr> at 0x1029a5fa0>
>>> next(g)
1
>>> next(g)
4
>>> iter(g)
<generator object <genexpr> at 0x1029a5fa0>
>>> iter(g) is g
True
>>> [g.next() for __ in xrange(10)]
[9, 16, 25, 36, 49, 64, 81, 100, 121, 144]
```  

# List Comprehensions

List comprehensions provide a concise way to create lists. It consists of brackets containing an expression followed by a for clause, then
zero or more for or if clauses. The expressions can be anything, meaning you can put in all kinds of objects in lists. The result will be a new list resulting from evaluating the expression in the
context of the for and if clauses which follow it. The list comprehension always returns a result list.

This
```python
new_list = []
for i in old_list:
    if filter(i):
        new_list.append(expressions(i))
```
and this are equivalent

```python
new_list = [expression(i) for i in old_list if filter(i)]
```

# Decorators  

A decorator is a design pattern in which a class or function alters or adds to the functionality of another class or function without using inheritance, or directly modifying the source code. In Python, decorators are, in simplest terms, functions (or any callable objects) that take as input a set of optional arguments and a function or class, and return a function or class.  

```Python  
def logged(time_format):
   def decorator(func):
      def decorated_func(*args, **kwargs):
         print "- Running '%s' on %s " % (
                                         func.__name__,
                                         time.strftime(time_format)
                              )
         start_time = time.time()
         result = func(*args, **kwargs)
         end_time = time.time()
         print "- Finished '%s', execution time = %0.3fs " % (
                                         func.__name__,
                                         end_time - start_time
                              )

         return result
     decorated_func.__name__ = func.__name__
     return decorated_func
 return decorator
 ```
 Here functions add1 and add2 are decorated using logged and a sample output is given.
 ```python
 @logged("%b %d %Y - %H:%M:%S")
def add1(x, y):
    time.sleep(1)
    return x + y


@logged("%b %d %Y - %H:%M:%S")
def add2(x, y):
    time.sleep(2)
    return x + y


print add1(1, 2)
print add2(1, 2)

# Output:
- Running 'add1' on Jul 24 2013 - 13:40:47
- Finished 'add1', execution time = 1.001s
3
- Running 'add2' on Jul 24 2013 - 13:40:48
- Finished 'add2', execution time = 2.001s
3
```
