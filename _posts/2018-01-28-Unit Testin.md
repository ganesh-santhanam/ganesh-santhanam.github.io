---
title: "Unit Testing"
date: 2018-01-28
mathjax: "true"
---
#Unit Testing

Unit testing involves breaking your program into pieces, and subjecting each piece to a series of tests.They should be done as often as possible. When you are performing tests as part of the development process, your code is automatically going to be designed better . Unit testing reduces the number of bugs released during deployment, making it critical to effective software development.
Python has multiple unit testing packages. Here we are going to use PyTest

 A simple test function using PyTest. Pytest will run all files of the form test_*.py or \*\_test.py in the current directory and its subdirectories.


 ```python
 def func(x):
    return x + 1

def test_answer():
    assert func(3) == 5
```

```
$ pytest
=========================== test session starts ============================
platform linux -- Python 3.x.y, pytest-4.x.y, py-1.x.y, pluggy-0.x.y
rootdir: $REGENDOC_TMPDIR, inifile:
collected 1 item

test_sample.py F                                                     [100%]

================================= FAILURES =================================
_______________________________ test_answer ________________________________

    def test_answer():
>       assert func(3) == 5
E       assert 4 == 5
E        +  where 4 = func(3)

test_sample.py:5: AssertionError
========================= 1 failed in 0.12 seconds =========================
```

#Assert that a certain exception is raised
```python
import pytest
def f():
    raise SystemExit(1)

def test_mytest():
    with pytest.raises(SystemExit):
```      
```
$ pytest -q test_sysexit.py
.                                                                    [100%]
1 passed in 0.12 seconds
```

#pytest fixtures
We can add specific code to run:

* at the beginning and end of a module of test code (setup_module/teardown_module)
* at the beginning and end of a class of test methods (setup_class/teardown_class)
* before and after a test function call (setup_function/teardown_function)
before and after a test method call (setup_method/teardown_method)

```python
ef setup_module(module):
    print ("setup_module      module:%s" % module.__name__)

def teardown_module(module):
    print ("teardown_module   module:%s" % module.__name__)

def setup_function(function):
    print ("setup_function    function:%s" % function.__name__)

def teardown_function(function):
    print ("teardown_function function:%s" % function.__name__)

def test_numbers_3_4():
    print 'test_numbers_3_4  <============================ actual test code'
    assert multiply(3,4) == 12

def test_strings_a_3():
    print 'test_strings_a_3  <============================ actual test code'
    assert multiply('a',3) == 'aaa'


class TestUM:

    def setup(self):
        print ("setup             class:TestStuff")

    def teardown(self):
        print ("teardown          class:TestStuff")

    def setup_class(cls):
        print ("setup_class       class:%s" % cls.__name__)

    def teardown_class(cls):
        print ("teardown_class    class:%s" % cls.__name__)

    def setup_method(self, method):
        print ("setup_method      method:%s" % method.__name__)

    def teardown_method(self, method):
        print ("teardown_method   method:%s" % method.__name__)

    def test_numbers_5_6(self):
        print 'test_numbers_5_6  <============================ actual test code'
        assert multiply(5,6) == 30

    def test_strings_b_2(self):
        print 'test_strings_b_2  <============================ actual test code'
        assert multiply('b',2) == 'bb'
```
```
> py.test -s test_um_pytest_fixtures.py
============================= test session starts ==============================
platform win32 -- Python 2.7.3 -- pytest-2.2.4
collecting ... collected 4 items

test_um_pytest_fixtures.py ....

=========================== 4 passed in 0.07 seconds ===========================
setup_module      module:test_um_pytest_fixtures
setup_function    function:test_numbers_3_4
test_numbers_3_4  <============================ actual test code
teardown_function function:test_numbers_3_4
setup_function    function:test_strings_a_3
test_strings_a_3  <============================ actual test code
teardown_function function:test_strings_a_3
setup_class       class:TestUM
setup_method      method:test_numbers_5_6
setup             class:TestStuff
test_numbers_5_6  <============================ actual test code
teardown          class:TestStuff
teardown_method   method:test_numbers_5_6
setup_method      method:test_strings_b_2
setup             class:TestStuff
test_strings_b_2  <============================ actual test code
teardown          class:TestStuff
teardown_method   method:test_strings_b_2
teardown_class    class:TestUM
teardown_module   module:test_um_pytest_fixtures
```        

# A summary of python code style conventions

“PEP 8: Style Guide for Python Code” and “PEP 257: Docstring Conventions” are the python style guide conventions. This is summarized here. They help in writing readable and expressive code.

# Indentation, line-length & code wrapping
* Always use 4 spaces for indentation (don’t use tabs)
* Write in ASCII in Python 2 and UTF-8 in Python 3
* Max line-length: 72 characters (especially in comments)
* Always indent wrapped code for readablility
```python
# Good:
result = some_function_that_takes_arguments(
    'argument one,
    'argument two',
    'argument three'
)

# Bad:
result = some_function_that_takes_arguments(
'argument one,
'argument two', 'argument three')
result2 = some_function_that_takes_arguments('argument one', 'argument two', 'argument three')
```

#Imports
* Don’t use wildcards
* Try to use absolute imports over relative ones
* When using relative imports, be explicit (with .)
* Don’t import multiple packages per line
```python
# Good:
import os
import sys
from mypkg.sibling import example
from subprocess import Popen, PIPE # Acceptable
from .sibling import example # Acceptable

# Bad:
import os, sys # multiple packages
import sibling # local module without "."
from mypkg import * # wildcards
```

#Whitespace and newlines
* 2 blank lines before top-level function and class definitions
* 1 blank line before class method definitions
* Use blank lines in functions sparingly
* Avoid extraneous whitespace
* Don’t use whitespace to line up assignment operators (=, :)
* Spaces around = for assignment
* No spaces around = for default parameter values
* Spaces around mathematical operators, but group them sensibly
* Multiple statements on the same line are discouraged
```python
# Good:
spam(ham[1], {eggs: 2})
if x == 4:
    print x, y
    x, y = y, x
dict['key'] = list[index]
y = 2
long_variable = 3
hypot2 = x*x + y*y
c = (a+b) * (a-b)
def complex(real, imag=0.0):
    return magic(r=real, i=imag)
do_one()
do_two()

# Bad
spam ( ham[ 1 ], { eggs: 2 } ) # spaces inside brackets
if x == 4 : print x , y ; x , y = y , x # inline statements, space before commas
dict ['key'] = list [index] # space before dictionary key
y             = 2 # Using spaces to line up assignment operators
long_variable = 3
hypot2 = x * x + y * y # Too much space around operators
c = (a + b) * (a - b) # Too much space around operators
def complex(real, imag = 0.0):
    return magic(r = real, i = imag) # Spaces in default values

```    
# Comments
* Keep comments up to date - incorrect comments are worse than no comments
* Write in whole sentences
* Use inline comments sparingly & avoid obvious comments
* Each line of block comments should start with “# “
* Paragraphs in block comments should be separated by a line with a single “#”
* All public functions, classes and methods should have docstrings
* Docstrings should start and end with """
* Docstring one-liners can be all on the same line
* In docstrings, list each argument on a separate line
* Docstrings should have a blank line before the final """
```python

def my_function():
    """ A one-line docstring """

def my_other_function(parameter=False):
    """
    A multiline docstring.

    Keyword arguments:
    parameter -- an example parameter (default False)

    """
```
#Naming conventions
* Class names in CapWords
* Method, function and variables names in lowercase_with_underscores
* Private methods and properties start with \__double_underscore
* “Protected” methods and properties start with \_single_underscore
* If you need to use a reserved word, add a _ to the end (e.g. class_)
* Always use self for the first argument to instance methods
* Always use cls for the first argument to class methods
* Never declare functions using lambda (f = lambda x: 2*x)
```python
class MyClass:
    """ A purely illustrative class """

    __property = None

    def __init__(self, property_value):
        self.__property = property_value

    def get_property(self):
        """ A simple getter for "property" """

        return self.__property

    @classmethod
    def default(cls):
        instance = MyClass("default value")
        return instance
  ```
