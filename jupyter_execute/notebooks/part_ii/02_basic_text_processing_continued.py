#!/usr/bin/env python
# coding: utf-8

# # Manipulating text at scale
# 
# This section introduces you to regular expressions for manipulating text and how to apply the same procedure to several files.
# 
# After reading this section, you should know:
# 
#  - how to manipulate multiple text files using Python
#  - how to define simple patterns using *regular expressions*
#  - how to save the results

# ## Scaling up
# 
# Ideally, Python should enable you to manipulate text at scale, that is, to apply the same procedure to ten, hundred or thousand text files *with the same effort*.
# 
# To do so, we must be able to define more flexible patterns than the fixed strings that we used previously with the `replace()` method, while opening and closing files automatically.
# 
# This capability is provided by Python modules for *regular expressions* and *file handling*.

# ## Regular expressions

# In[1]:


# Run this cell to view a YouTube video related to this topic
from IPython.display import YouTubeVideo
YouTubeVideo('seCpHdTA-vs', height=350, width=600)


# [Regular expressions](https://en.wikipedia.org/wiki/Regular_expression) are a "language" that allows defining *search patterns*.
# 
# These patterns can be used to find or to find and replace patterns in Python string objects.
# 
# As opposed to fixed strings, regular expressions allow defining *wildcard characters* that stand in for any character, *quantifiers* that match sequences of repeated characters, and much more.
# 
# Python allows using regular expressions through its `re` module.
# 
# We can activate this module using the `import` command.

# In[2]:


import re


# Let's begin by loading the text file, reading its contents, assigning the last 2000 characters to the variable `extract` and printing out the result.

# In[3]:


# Define a path to the file, open the file for (r)eading using utf-8 encoding
file = open(file='data/WP_1990-08-10-25A.txt', mode='r', encoding='utf-8')

# Read the file contents using the .read() method
text = file.read()

# Get the *last* 2000 characters – note the minus sign before the number
extract = text[-2000:]

# Print the result
print(extract)


# As you can see, the text has a lot of errors from optical character recognition, mainly in the form of sequences such as `....` and `,,,,`.
# 
# Let's compile our first regular expression that searches for sequences of *two or more* full stops.
# 
# This is done using the `compile()` function from the `re` module.
# 
# The `compile()` function takes a string as an input. 
# 
# Note that we attach the prefix `r` to the string. This tells Python to store the string in 'raw' format. This means that the string is stored as it appears.

# In[4]:


# Compile a regular expression and assign it to the variable 'stops'
stops = re.compile(r'\.{2,}')

# Let's check the type of the regular expression!
type(stops)


# Let's unpack this regular expression a bit.
# 
# 1. The regular expression is defined using a Python string, as indicated by the single quotation marks `'  '`.
# 
# 2. We need a backslash `\` in front of our full stop `.`. The backslash tells Python that we are really referring to a full stop, because regular expressions use a full stop as a *wildcard* character that can stand in for *any character*.
# 
# 3. The curly brackets `{ }` instruct the regular expression to search for instances of the previous item `\.` (our actual full stop) that occur two or more times (`2,`). This (hopefully) preserves true uses of a full stop!
# 
# In plain language, we tell the regular expression to search for *occurrences of two or more full stops*. 

# To apply this regular expression to some text, we will use the `sub()` method of our newly-defined regular expression object `stops`.
# 
# The `sub()` method takes two arguments:
# 
# 1. `repl`: A string containing a string that is used to *replace* possible matches.
# 2. `string`: A string object to be searched for matches.
# 
# The method returns the modified string object.
# 
# Let's apply our regular expression to the string stored under the variable `extract`.

# In[5]:


# Apply the regular expression to the text under 'extract' and save the output
# to the same variable, essentially overwriting the old text.
extract = stops.sub(repl='', string=extract)

# Print the text to examine the result
print(extract)


# As you can see, the sequences of full stops are gone.
# 
# We can make our regular expression even more powerful by adding alternatives.
# 
# Let's compile another regular expression and store it under the variable `punct`.

# In[6]:


# Compile a regular expression and assign it to the variable 'punct'
punct = re.compile(r'(\.|,){2,}')


# What's new here are the parentheses `( )` and the vertical bar `|` between them, which separates our actual full stop `\.` and the comma `,`.
# 
# The characters surrounded by parentheses and separated by a vertical bar mark *alternatives*.
# 
# In plain English, we tell the regular expression to search for *occurrences of two or more full stops or commas*.
# 
# Let's apply our new pattern to the text under `extract`.
# 
# To ensure the pattern works as intended, let's retrieve the original text from the `text` variable and assign it to the variable `extract` to overwrite our previous edits.

# In[7]:


# "Reset" the extract variable by taking the last 2000 characters of the original string
extract = text[-2000:]

# Apply the regular expression
extract = punct.sub(repl='', string=extract)

# Print out the result
print(extract)


# Success! The sequences of full stops and commas can be removed using a single regular expression.

# ### Quick exercise
# 
# Use `re.compile()` to compile a regular expression that matches `”`, `""` and `’’` and store the result under the variable `quotes`.
# 
# Find matching sequences in `extract` and replace them with `"`.
# 
# You will need parentheses `( )` and vertical bars `|` to define the alternatives.

# In[8]:


### Enter your code below this line and run the cell (press Shift and Enter at the same time)


# The more irregular sequences resulting from optical character recognition errors in `extract`, such as `'-'*`, `->."`, `/*—.`, `-"“` and `'"''.` are much harder to capture.
# 
# Capturing these patterns would require defining more complex regular expressions, which are harder to write. Their complexity is, however, what makes regular expressions so powerful, but at the same time, learning how to use them takes time and patience.
# 
# It is therefore a good idea to use a service such as [regex101.com](https://www.regex101.com) to learn the basics of regular expressions.
# 
# In practice, coming up with regular expressions that cover as many matches as possible is particularly hard. 
# 
# Capturing most of the errors – and perhaps distributing the manipulations over a series of steps in a pipeline – can already help prepare the text for further processing or analysis.
# 
# However, keep in mind that in order to identify patterns for manipulating text programmatically, you should always look at more than one text in your corpus.

# ## Processing multiple files

# In[9]:


# Run this cell to view a YouTube video related to this topic
from IPython.display import YouTubeVideo
YouTubeVideo('UYyqQD3w59c', height=350, width=600)


# Many corpora contain texts in multiple files. 
# 
# To make manipulating the text as efficient as possible, we must open the files, read their contents, perform the requested operations and close them *programmatically*.
# 
# This procedure is fairly simple using the `Path` class from Python's `pathlib` module.
# 
# Let's import the class first. Using the command `from` with `import` allows us to import only a part of the `pathlib` module, namely the `Path` class. This is useful if you only need some feature contained in a Python module or library.

# In[10]:


from pathlib import Path


# The `Path` class encodes information about *paths* in a *directory structure*.
# 
# What's particularly great about the Path class is that it can automatically infer what kinds of paths your operating system uses. 
# 
# Here the problem is that operating systems such as Windows, Linux and Mac OS X have different file system paths.
# 
# Using the `Path` class allows us to avoid a lot of trouble, particularly if we want to code to run on different operating systems.
# 
# Our repository contains a directory named `data`, which contains the text files that we have been working with recently.
# 
# Let's initialise a Path *object* that points towards this directory by providing a string with the directory name to the Path *class*. We assign the object to the variable `corpus_dir`.

# In[11]:


# Create a Path object that points towards the directory 'data' and assign
# the object to the variable 'corpus_dir'
corpus_dir = Path('data')


# The Path object stored under `corpus_dir` has various useful methods and attributes.
# 
# We can, for instance, easily check if the path is valid using the `exists()` method.

# In[12]:


# Use the exists() method to check if the path is valid
corpus_dir.exists()


# We can also check if the path is a directory using the `is_dir()` method.

# In[13]:


# Use the exists() method to check if the path points to a directory
corpus_dir.is_dir()


# Let's make sure the path does not point towards a file using the `is_file()` method.

# In[14]:


# Use the exists() method to check if the path points to a file
corpus_dir.is_file()


# Now that we know that the path referred to is indeed a directory, we can use the `glob()` method to collect all text files in the directory.
# 
# `glob` stands for [*global*](https://en.wikipedia.org/wiki/Glob_(programming)), and was first implemented as a program for matching filenames and paths using wildcards.
# 
# The `glob()` method requires one argument, `pattern`, which takes a string as input. This string defines the kinds of files to be collected. The asterisk symbol `*` acts as a wildcard, which can refer to *any sequence of characters* preceding the sequence `.txt`.
# 
# The file identifier `.txt` is a commonly-used suffix for plain text files.
# 
# We also instruct Python to *cast* the result into a list using the `list()` function, so we can easily loop over the files in the list.
# 
# Finally, we store the result under the variable `files` and call the result.

# In[15]:


# Get all files with the suffix .txt in the directory 'corpus_dir' and cast the result into a list
files = list(corpus_dir.glob(pattern='*.txt'))

# Call the result
files


# We now have a list of three Path objects that point towards three text files!
# 
# This allows us to loop over the files using a `for` loop and manipulate text in each file.

# In[16]:


# Begin the loop
for f in files:
    
    # Open the file at the path stored under the variable 'f' and declare encoding ('utf-8')
    file = open(f, encoding="utf-8")
    
    # The Path object contains the filename under the attribute name. Let's assign the filename to a variable.
    filename = f.name
    
    # Read the file contents
    text = file.read()
    
    # Print the filename and the first 100 characters in the file
    print(filename, text[:100])
    
    # Define a new filename for our modified file, which has the prefix 'mod_'
    new_filename = 'mod_' + filename
    
    # Define a new path. Path will automatically join the directory and filenames for us.
    new_path = Path('data', new_filename)
    
    # We then create a file with the new filename. Note the mode for *writing*
    new_file = open(new_path, mode='w+', encoding="utf-8")
    
    # Apply our regular expression for removing excessive punctuation to the text
    modified_text = punct.sub('', text)
    
    # Write the modified text to the new file
    new_file.write(modified_text)
    
    # Let's close the files
    file.close()
    new_file.close()


# If you take a look at the directory [data](data), you should now see three files whose names have the prefix `mod_`. These are the files we just modified and saved.
# 
# To keep the data directory clean, run the following cell to delete the modified files.
# 
# Adding the exclamation mark `!` to the beginning of a code cell tells Jupyter that this is a command to the underlying command line interface, which can be used to manipulate the file system.
# 
# In this case, we run the command `rm` to delete all files in the directory `data`, whose filename begins with the characters `mod`.

# In[17]:


get_ipython().system('rm data/mod*')


# This should have given you an idea of the some more powerful methods for manipulating text available in Python, such as regular expressions, and how to apply them to multiple files at the same time.
# 
# The [following section](03_basic_nlp.ipynb) will teach you how to apply basic natural language processing techniques to texts.
