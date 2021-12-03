#!/usr/bin/env python
# coding: utf-8

# # The elements of a Jupyter Notebook

# In[1]:


# Run this cell to view a YouTube video related to this topic
from IPython.display import YouTubeVideo
YouTubeVideo('cthzk6B80ds', height=350, width=600)


# Jupyter Notebooks are made up of cells, which contain either content or code. 
# 
# Content cells are written in Markdown, which is a markup language for formatting content.
# 
# Code cells, in turn, contain code, which in our case is written in Python 3.

# To see the difference between Markdown and code cells in Jupyter Notebook, run the cells below.
# 
# To run a cell, press the _Run_ button in the toolbar on top of the Jupyter Notebook or press the <kbd>Shift</kbd> and <kbd>Enter</kbd> keys on your keyboard at the same time.
This is a markdown cell.
# In[2]:


print("This is a code cell.")


# As you can see, running the cell moves the *cursor* (indicated by the coloured bounding box around the cell) to the next cell.

# The code cells, such as the one above, are typically run one by one, while documenting and describing the process using the content cells. 
# 
# You can also run all cells in a notebook by choosing _Run All_ in the _Cell_ menu on top of the Jupyter Notebook. Press <kbd>H</kbd> on your keyboard for a list of shortcuts for various commands.
# 
# The number in brackets on the left-hand side of a cell indicates the order in which cells have been executed.
# 
# In most cases, cells must be run in a sequential order for the program to work.
