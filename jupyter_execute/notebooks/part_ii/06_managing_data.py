#!/usr/bin/env python
# coding: utf-8

# # Managing textual data using pandas
# 
# This section introduces how to prepare and manage textual data for analysis using [pandas](http://pandas.pydata.org/), a Python library for working with tabular data.
# 
# After reading this section, you should know:
# 
# - how to import data into a pandas DataFrame
# - how to explore data stored in a pandas DataFrame
# - how to append data to a pandas DataFrame
# - how to save the data in a pandas DataFrame

# ## Importing data to pandas

# In[1]:


# Run this cell to view a YouTube video related to this topic
from IPython.display import YouTubeVideo
YouTubeVideo('UQYyCVGZbvk', height=350, width=600)


# Let's start by importing the pandas library. 
# 
# Note that we can control the name of the imported module using the `as` addition to the `import` command. pandas is commonly abbreviated `pd`.
# 
# This allows us to use the variable `pd` to refer to the *pandas* library.

# In[2]:


import pandas as pd


# ### Importing data from a single file
# 
# You must often load and prepare the data yourself, either from a single file or from multiple files.
# 
# Typical formats for distributing corpora include CSV files, which stands for Comma-separated Values, and JSON, which stands for JavaScript Object Notation or simple plain text files.
# 
# pandas provides plenty of functions for [reading data in various formats](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html). You can even try importing Excel sheets!
# 
# The following example shows how to load a corpus from a CSV file for processing in Python using the [SFU Opinion and Comments Corpus (SOCC)](https://github.com/sfu-discourse-lab/SOCC) (Kolhatkar et al. [2020](https://doi.org/10.1007/s41701-019-00065-w)).
# 
# Let's load a part of the SFU Opinion and Comments Corpus, which contains the opinion articles from [The Globe and Mail](https://www.theglobeandmail.com/), a Canadian newspaper.
# 
# We can use the `read_csv()` function from pandas to read files with comma-separated values, such as the SOCC corpus.
# 
# The `read_csv()` function takes a string object as input, which defines a path to the input file.

# In[3]:


# Read the CSV file and assign the output to the variable 'socc'
socc = pd.read_csv('data/socc_gnm_articles.csv')


# pandas does all the heavy lifting and returns the contents of the CSV file in a pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), which is data structure native to pandas.

# In[4]:


# Examine the type of the object stored under the variable 'socc'
type(socc)


# Let's use the `head()` method of a DataFrame to check out the first five rows of the DataFrame.

# In[5]:


# Print the first five rows of the DataFrame
socc.head(5)


# As you can see, the DataFrame has a tabular form.
# 
# The DataFrame contains several columns such as **article_id**, **title** and **article_text**, accompanied by an index for each row (**0, 1, 2, 3, 4**).
# 
# The `.at[]` accessor can be used to inspect a single item in the DataFrame.
# 
# Let's examine the value in the column **title** at index 123.

# In[6]:


socc.at[123, 'title']


# ### Quick exercise
# 
# Let's go back to the SOCC corpus stored under the variable `socc`.
# 
# Who is the author (`author`) of article at index 256? 
# 
# How many top-level comments (`ntop_level_comments`) did the article at index 1000 receive?

# In[7]:


### Enter your code below this line and run the cell (press Shift and Enter at the same time)


# ### Importing data from multiple files
# 
# Another common scenario is that you have multiple files with text data, which you want to load into *pandas*.
# 
# Let's first collect the files that we want to load.

# In[8]:


# Import the patch library
from pathlib import Path

# Create a Path object that points to the directory with data
corpus_dir = Path('data')

# Get all .txt files in the corpus directory
corpus_files = list(corpus_dir.glob('*.txt'))

# Check the corpus files
corpus_files


# To accommodate our data, let's create an empty pandas DataFrame and specify its *shape* in advance, that is, the number of rows (`index`) and the names of the columns `columns`.
# 
# We can determine the number of rows needed using Python's `range()` function. This function generates a list of numbers that fall within certain range, which we can use for the index of the DataFrame.
# 
# In this case, we define a `range()` between `0` and the number of text files in the directory, which are stored under the variable `corpus_files`. We retrieve their number using the `len()` function, which returns the length of Python objects, if applicable.
# 
# For the columns of the DataFrame, we simply create columns for filenames and their textual content by providing a list of strings to the `columns` argument.

# In[9]:


# Create a DataFrame and assign the result to the variable 'df'
df = pd.DataFrame(index=range(0, len(corpus_files)), columns=['filename', 'text'])

# Call the variable to inspect the output
df


# Now that we have an empty data with rows for each file in the corpus, we can loop over the file paths under `corpus_files` and add their contents to the DataFrame.

# In[10]:


# Loop over the corpus files and count each loop using enumerate()
for i, f in enumerate(corpus_files):
    
    # Open the file for reading
    c_file = open(f, encoding="utf-8")
    
    # Get the filename from the Path object
    filename = f.name
        
    # Read the file contents
    text = c_file.read()
    
    # Assign the text from the file to index 'i' at column 'text'
    # using the .at accessor – note that this modifies the DataFrame
    # "in place" – you don't need to assign the result into a variable
    df.at[i, 'text'] = text
    
    # We then do the same to the filename
    df.at[i, 'filename'] = filename


# Let's check the result by calling the variable `df`.

# In[11]:


# Call the variable to check the output
df


# As you can see, the DataFrame has been populated with filenames and text.
# 
# Now that we know how to load data into DataFrames, we can turn towards accessing and manipulating the data that they store.

# ## Examining DataFrames

# In[12]:


# Run this cell to view a YouTube video related to this topic
from IPython.display import YouTubeVideo
YouTubeVideo('-zhW3YjhuzI', height=350, width=600)


# ## Examining DataFrames
# 
# pandas DataFrames can hold a lot of information, which is often organised into columns.
# 
# The columns present in a DataFrame are accessible through the attribute `columns`.

# In[13]:


# Retrieve the columns and their names
socc.columns


# pandas provides various methods for examining the contents of entire columns, which can be accessed just like the keys and values of a Python dictionary.
# 
# The brackets `[]` can be used to access entire columns by placing the column name within the brackets as a string.
# 
# Let's retrieve the contents of the column `author`, which contains author information.

# In[14]:


# Retrieve the contents of the column 'author' in the DataFrame 'socc'
socc['author']


# As you can see, the column `author` contains 10399 objects, as indicated by the *Length* and *dtype* properties. The numbers on the left-hand side give the index, that is, the row numbers.
# 
# The columns of a pandas DataFrame consist of another object type, namely *pandas* Series. You can think of the DataFrame as an entire table, whose columns consist of Series.
# 
# We can verify this by examining their type using Python's `type()` function.

# In[15]:


# Check the type of 'socc' and 'socc['author']'
type(socc), type(socc['author'])


# When printing out the contents of a DataFrame or Series, pandas omits everything between the first and last five rows by default. This is convenient when working with thousands of rows.
# 
# This also applies to the output for methods such as `value_counts()`, which allows counting the number of unique values in a Series.

# In[16]:


# Count unique values in the column 'author'
socc['author'].value_counts()


# Not surprisingly, the editorial team at the *The Globe and Mail* is responsible for most of the editorials!
# 
# Let's take another look at the data by visualising the result by calling the `.plot()` method for the author information column. This method calls an external library named matplotlib, which can be used to produce all kinds of plots and visualisations.
# 
# More specifically, we instruct the `plot()` method to draw a bar chart by providing the string `bar` to the `kind` argument.
# 
# We also use the brackets `[:10]` to limit the output to the ten most profilic authors.

# In[17]:


# This is some Jupyter magic that allows us to render matplotlib plots in the notebooks!
# You only need to enter this command once.
get_ipython().run_line_magic('matplotlib', 'inline')

# Count the values in the column 'author' and clip the result to top-10 before plotting.
socc['author'].value_counts()[:10].plot(kind='bar')


# For columns with numerical values, we can also use the `describe()` method to get basic descriptive statistics on the data.

# In[18]:


# Get basic descriptive statistics for the column 'ntop_level_comments'
socc['ntop_level_comments'].describe()


# As we can see, the column `ntop_level_comments` has a total of 10339 rows. 
# 
# The average number of comments received by an editorial is approximately 26, but this number fluctuates, as the standard deviation from the average is nearly 40.
# 
#  - Some editorials do not have any comments at all, as indicated by the minimum value of 0. 
#  - The lowest quartile shows that 25% of the data has only one comment or less (none).
#  - The second quartile (50%), which is also known as the median, indicates that *half* of the data has less than 14 comments and *half* has more than 14 comments. 
#  - The third quartile shows that 75% of the data has 35 comments or less. 
#  - The most commented editorial has 1378 comments.

# What if we would like to find the articles with zero comments?
# 
# We can use the DataFrame accessor `.loc` to access specific rows based on their values.
# 
# The number of comments is stored in the column `ntop_level_comments`, but we also need to specify that the DataFrame stored under the variable `socc` contains the column that we wish to examine. 
# 
# This causes the somewhat repetitive reference to the `socc` DataFrame, which is nevertheless necessary for being explicit.
# 
# We also need to provide a command for "is equal to". Since the single equal sign `=` is reserved for assigning variables in Python, two equal signs `==` are used for comparison.
# 
# Finally, we place the value we want to evaluate against on the right-hand side of the double equal sign `==`, that is, zero for no comments.

# In[19]:


# Get rows with no top level comments
socc.loc[socc['ntop_level_comments'] == 0]


# This returns a total of 2542 rows where the value of the column `ntop_level_comments` is zero.
# 
# For more complex views of the data, we can also combine multiple criteria using the `&` symbol, which is the Python operator for "AND".
# 
# Note that individual criteria must be placed in parentheses `()` to perform the operation.
# 
# Let's check if the first author in our result, Hayden King, wrote any other articles with zero comments.

# In[20]:


# Get number of top level comments for author Hayden King
socc.loc[(socc['ntop_level_comments'] == 0) & (socc['author'] == 'Hayden King')]


# ### Quick in-class exercise
# 
# How many articles with zero top-level comments were authored by the editorial team (`GLOBE EDITORIAL`)?
# 
# Write out the whole command yourself instead of copy-pasting to get an idea of the syntax.

# In[21]:


### Enter your code below this line and run the cell (press Shift and Enter at the same time)


# ## Extending DataFrames

# In[22]:


# Run this cell to view a YouTube video related to this topic
from IPython.display import YouTubeVideo
YouTubeVideo('Yj87Gy28vqs', height=350, width=600)


# You can easily add information to pandas DataFrames.
# 
# One common scenario could involve loading some data from an external file (such as a CSV or JSON file), performing some analyses and storing the results to the same DataFrame.
# 
# We can easily add an empty column to the DataFrame. This is achieved using the column accessor `[]` and the Python datatype `None`.
# 
# Let's add a new column named `comments_ratio` to the DataFrame `socc`.

# In[23]:


# Add a new column named 'comments_ratio' to the DataFrame
socc['comments_ratio'] = None


# In[24]:


# Print out the first five rows of the DataFrame
socc.head(5)


# Let's populate the column with some data by calculating which percentage of the comments are top-level comments, assuming that a high percentage of top-level comments indicates comments about the article, whereas a lower percentage indicates more discussion about the comments posted.
# 
# To get the proportion of top-level comments out of all comments, we must divide the number of top-level comments by the number of all comments.

# In[25]:


# Populate the 'comments_ratio' column by calculating the ratio of top-level comments and comments
socc['comments_ratio'] = socc['ntop_level_comments'] / socc['ncomments']


# Column accessors can be used very flexibly to access and manipulate data stored in the DataFrame, as exemplified by the division.

# In[26]:


# Print out the first five rows of the DataFrame
socc.head(5)


# As you can see, the column `comments_ratio` now stores the result of our calculation!
# 
# However, we should also keep in mind that some articles did not receive any comments at all: thus we would have divided zero by zero.
# 
# Let's examine these cases again by retrieving articles without comments, and use the `head()` method to limit the output to the first five rows.

# In[27]:


# Print out the first five comments with no top-level comments
socc.loc[socc['ntop_level_comments'] == 0].head(5)


# For these rows, the `comments_ratio` column contains values marked as `NaN` or "not a number".
# 
# This indicates that the division was performed on these cells as well, but the result was not a number.
# 
# *pandas* automatically ignores `NaN` values when performing calculations, as show by the `.describe()` method.

# In[28]:


# Get descriptive statistics for the column 'comments_ratio'
socc['comments_ratio'].describe()


# Note the difference in the result for the count. Only 7797 items out of 10399 were included in the calculation.

# What if we would like to do some natural language processing and store the results in the DataFrame?
# 
# Let's select articles that fall within the first quartile in terms of the ratio of original comments to all comments made (`comments_ratio`) and have received more than 200 comments (`ncomments`). 

# In[29]:


# Filter the DataFrame for highly commented articles and assign the result to the variable 'talk'
talk = socc.loc[(socc['comments_ratio'] <= 0.384) & (socc['ncomments'] >= 200)]

# Call the variable to examine the output
talk.head(5)


# Let's import spaCy, load a medium-sized language model for English and assign this model to the variable `nlp`.

# In[30]:


# Import the spaCy library
import spacy

# Note that we now load a medium-sized language model!
nlp = spacy.load('en_core_web_md')


# Let's limit processing to article titles and create a placeholder column to the DataFrame named `processed_title`.

# In[31]:


# Create a new column named 'processed_title'
talk['processed_title'] = None


# pandas warns about performing this command, because `talk` is only a slice or a _view_ into the DataFrame. 
# 
# Assigning a new column to **only a part of the DataFrame** would cause problems by breaking the tabular structure.
# 
# We can fix the situation by creating a _deep copy_ of the slice using Python's `.copy()` method.

# In[32]:


# Create a deep copy of the DataFrame
talk = talk.copy()


# Let's try creating an empty column again.

# In[33]:


# Create a new column named 'processed_title'
talk['processed_title'] = None


# In[34]:


# Print out the first five rows of the DataFrame
talk.head(5)


# To retrieve the title for each article from the column `title`, feed it to the language model under `nlp` for processing and store the output into the column `processed_title`, we need to use the `apply()` method of a DataFrame.
# 
# As the name suggests, the `apply()` method applies whatever is provided as input to the method to each row in the column.
# 
# In this case, we pass the language model `nlp` to the `apply()` method, essentially retrieving the titles stored as string objects in the column `title` and "applying" the language model `nlp` to them.
# 
# We assign the output to the DataFrame column named `processed_title`.

# In[35]:


# Apply the language model under 'nlp' to the contents of the DataFrame column 'title'
talk['processed_title'] = talk['title'].apply(nlp)

# Call the variable to check the output
talk


# We now have the processed titles in a separate column named `processed_title`!
# 
# Let's examine the first row in the DataFrame `talk`, whose index is 2.

# In[36]:


# Get the value in the column 'processed_title' at row with index 2
talk.at[2, 'processed_title']


# In[37]:


# Check the type of the contained object
type(talk.at[2, 'processed_title'])


# As you can see, the cell contains a spaCy _Doc_ object.
# 
# Let's now define our own Python **function** to fetch lemmas for each noun in the title.
# 
# Python functions are _defined_ using the command `def`, which is followed by the name of the function, in this case `get_nouns`. 
# 
# The input to the function is given in parentheses that follow the name of the function.
# 
# In this case, we name a variable for the input called `nlp_text`. This is an arbitrary variable, which is needed for referring to whatever is being provided as input to the function. To put it simply, you can think of this variable as referring to any input that will be eventually provided to the function.

# In[38]:


# Define a function named 'get_nouns' that takes a single object as input.
# We refer to this input using the variable name 'nlp_text'.
def get_nouns(nlp_text):
    
    # First we make sure that the input is of correct type
    # by using the assert command to check the input type
    assert type(nlp_text) == spacy.tokens.doc.Doc
    
    # Let's set up a placeholder list for our lemmas
    lemmas = []
    
    # We begin then begin looping over the Doc object
    for token in nlp_text:
        
        # If the fine-grained POS tag for the token is a noun (NN)
        if token.tag_ == 'NN':
            
            # Append the token lemma to the list of lemmas
            lemmas.append(token.lemma_)
            
    # When the loop is complete, return the list of lemmas
    return lemmas


# Now that we have defined our function, we can use the function with the `.apply()` method to collect all nouns to the column `nouns`.

# In[39]:


# Apply the 'get_nouns' function to the column 'processed_title'
talk['nouns'] = talk['processed_title'].apply(get_nouns)

# Call the variable to examine the output
talk


# As you can see, an empty DataFrame column is actually not required for adding new data, because pandas creates a new column automatically through assignment, as exemplified by `talk['nouns']`.

# We can also easily extract information from DataFrames into Python's native data structures. 
# 
# The `tolist()` method, for instance, can be used to extract the contents of a *pandas* Series into a list.

# In[40]:


# Cast pandas Series to a list
noun_list = talk['nouns'].tolist()

# Call the variable to check the output
noun_list[:10]


# What we have now under `noun_list` is a list of lists, because each row in the `nouns` column contains a list.  
# 
# Let's loop over the list and collect the items into a single list named `final_list` using the `extend()` method of a Python list.

# In[41]:


# Set up the placeholder list
final_list = []

# Loop over each list in the list of lists
for nlist in noun_list:
    
    # Extend the final list with the current list
    final_list.extend(nlist)


# Let's briefly examine the first ten items in final list and then count the number of items in the list.

# In[42]:


# Print the first ten items in the list
final_list[:10]


# In[43]:


# Check the length of the list
len(final_list)


# To plot the 10 most frequent nouns, we can cast the `final_list` into a pandas Series, count the occurrences of each lemma using `value_counts()` and plot the result using the `plot()` method.

# In[44]:


# Convert the list into a pandas Series, count unique nouns
# using the value_counts() method, get the 10 most frequent
# items [:10] and plot the result into a bar chart using the
# plot() method and its attribute 'kind'.
pd.Series(final_list).value_counts()[:10].plot(kind='bar')


# ## Saving DataFrames

# In[45]:


# Run this cell to view a YouTube video related to this topic
from IPython.display import YouTubeVideo
YouTubeVideo('yK_3QgOVXIo', height=350, width=600)


# pandas DataFrames can be easily saved as *pickled* objects using the `to_pickle()` method.
# 
# The `to_pickle()` method takes a string as input, which defines a path to the file in which the DataFrame should be written.
# 
# Let's pickle the DataFrame with the three articles stored under `df` into a file named `pickled_df.pkl` into the directory `data`.

# In[46]:


# Write the DataFrame to disk using pickle
df.to_pickle('data/pickled_df.pkl')


# We can easily check if the data has been saved successfully by reading the file contents using the `read_pickle()` method.

# In[47]:


# Read the pickled DataFrame and assign the result to 'df_2'
df_2 = pd.read_pickle('data/pickled_df.pkl')

# Call the variable to examine the output
df_2


# Let's compare the DataFrames, which returns a Boolean value (True/False) for each cell.

# In[48]:


# Compare DataFrames 'df' and 'df_2'
df == df_2


# This section should have given you a basic idea of the *pandas* library and how DataFrames can be used to store and manipulate textual data.
# 
# In [Part III](../part_iii/01_multilingual_nlp.ipynb), you will learn more about natural language processing techniques and their applications.
