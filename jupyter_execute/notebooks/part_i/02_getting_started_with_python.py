#!/usr/bin/env python
# coding: utf-8

# # Getting started with Python

# In[1]:


# Run this cell to view a YouTube video related to this topic
from IPython.display import YouTubeVideo
YouTubeVideo('65u7GK9c78o', height=350, width=600)


# This brief introduction will give you an idea of Python syntax. 
# 
# You will learn about key concepts such as variables, what they are, and how they are created and updated. 
# 
# You will also learn about various types of objects defined in Python and how the type of an object determines its behaviour.

# ## Variables
# 
# Understanding the concept of a variable is crucial when getting started with Python and other programming languages.
# 
# To put it simply, variables are unique names for objects defined in the program. If an object does not have a name, it cannot be referred to elsewhere in the program.
# 
# In Python, variables are assigned on the fly using a single equal sign `=`. 
# 
# The name of the variable is positioned left of the equal sign, while the object that the variable refers to is placed on the right-hand side.
# 
# Let's create a variable named `var` containing a _string_ object and call this object by its name.
# 
# Note that string objects are always surrounded by single or double quotation marks!

# In[2]:


var = "This is a variable."


# The following cell simply calls the variable, returning the object that the variable refers to.

# In[3]:


var


# As the notion of a *variable* suggests, the value of a variable can be changed or updated.

# In[4]:


var = "Yes, the variable name stays the same but the contents change."


# In[5]:


var


# If you happen to need a placeholder for some object, you can also assign the value `None` to a variable.

# In[6]:


var = None


# In[7]:


var


# Variable names can be chosen freely and thus the names should be informative. 
# 
# Variable names are case sensitive, which means that `var` and `Var` are interpreted as different variables.

# In[8]:


Var


# Calling the variable `Var` raises a `NameError`, because a variable with this name has not been defined.

# Naming variables is only limited by keywords that are part of Python's syntax. 
# 
# Running the following cell prints out these keywords.

# In[9]:


import keyword

keywords = keyword.kwlist

print(keywords)


# Printing out a list of keywords introduces several important aspects of Python: the `import` command can be used to load additional modules and make their functionalities available in Python. 
# 
# We will frequently use the `import` command to import various external libraries and/or their parts for natural language processing and other tasks.
# 
# In this case, the _module_ `keyword` has an _attribute_ called `kwlist`, which contains a _list_ of keywords. We assign this list to the variable `keywords` and print out its contents using the `print()` _function_.

# ### Quick exercise
# 
# Choose a name for a variable and assign a string object that contains some text to the variable. Remember the quotation marks around string objects!

# In[10]:


### Enter your code below this line and run the cell (press Shift and Enter at the same time)


# ## Objects
# 
# A list is just one _type_ of object defined in Python. More specifically, a list is one kind of _data structure_ in Python.
# 
# We can use the `type()` _function_ to check the type of an object. To get the type of an object assigned to some variable, place its name within parentheses.

# In[11]:


type(keywords)


# Remember our variable `var`? Let's check its type as well.

# In[12]:


type(var)


# The `type()` function is essential when hunting for errors in code.
# 
# Knowing the type of a Python object is useful, because it determines what can be done with the object. 
# 
# For instance, brackets that follow the variable name can be used to access _items_ contained in a _list_. 
# 
# Note that Python lists are zero-indexed, which means that counting starts from zero, not one.

# In[13]:


keywords[3]


# This returns the fourth item in the `keywords` list. 
# 
# Can we do the same with the variable `var`?

# In[14]:


var[3]


# This will not work, since we set `var` to `None`, which is a special type of object called _NoneType_.
# 
# Python raises a `TypeError`, because unlike a _list_ object, a _NoneType_ object cannot contain any other objects.
# 
# Let's return to the list of Python keywords under the variable `keywords` and check the type of the fourth _item_ in the _list_.

# In[15]:


type(keywords[3])


# As you can see, a _list_ can contain other types of objects.
# 
# Both strings and lists are common types when working with textual data.
# 
# Let's define a toy example consisting of a string with some HTML (Hypertext Markup Language, the language used for creating webpages) tags.

# In[16]:


text = "<p>This is an <b>example</b> string with some HTML tags thrown in.</p>"


# In[17]:


text


# Python provides various methods for manipulating strings such as the one stored under the variable `text`. 
# 
# The `split()` method, for instance, splits a _string_ into a _list_.
# 
# The `sep` argument defines the character that is used as the boundary for a split. 
# 
# By default, the separator is a _whitespace_ or empty space.
# 
# Let's use the `split()` method to split the string under `text` at empty space.

# In[18]:


tokens = text.split(sep=' ')


# We assign the result to the varible `tokens`. Calling the variable returns a list.

# In[19]:


tokens


# We can just as easily define some other separator, such as the less than symbol (<) marking the beginning of an HTML tag.

# In[20]:


text.split('<')


# As you can see, the `split()` method is destructive: the character that we defined as the boundary is deleted from each string in the list.
# 
# Note that we do not necessarily have to give the arguments such as `sep` explicitly: a correct type (string, `':'`) at the correct position (as the first *argument*) is enough.

# What if we would like to remove the HTML tags from our example string?
# 
# Let's go back to our original string stored under the variable `text`.

# In[21]:


text


# Python strings also have a `replace()` method, which allows replacing specific characters or their sequences in a string.
# 
# Let's begin by replacing the initial tag `<p>` in `text` by providing `'<p>'` as input to its `replace` method.
# 
# Note that the tag `<p>` is in quotation marks, as the `replace` method requires the input to be a string.
# 
# The `replace` method takes two inputs: the string to be replaced (`<p>`) and the replacement (`''`). By providing an empty string as input to the second argument, we essentially remove any matches from the string.

# In[22]:


text = text.replace('<p>', '')


# In[23]:


text


# Success! The first tag `<p>` is no longer present in the string. The other strings, however, remain in place.

# ### Quick exercise
# 
# What about the remaining tags? Replace the `<b>` tag in `text` with an empty string.

# In[24]:


### Enter your code below this line and run the cell (press Shift and Enter at the same time)


# Although the `replace` method allowed us to easily replace parts of a string, it is not the most effective way to do so. What if the data contains dozens of HTML tags or other kind of markup? For this reason, we will explore more efficient ways of manipulating text data in Part II.
# 
# This introduction should have given you a first taste of Python and its syntax. We will continue to learn more Python while working with actual examples.
