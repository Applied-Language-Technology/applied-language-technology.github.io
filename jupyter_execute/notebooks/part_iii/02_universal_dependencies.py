#!/usr/bin/env python
# coding: utf-8

# # Universal Dependencies

# > ⚠️ The cells of this notebook have been executed to facilitate the use of [readthedocs.io](applied-language-technology.readthedocs.io/). If you wish to work through the notebook step-by-step, go to the *Kernel* menu and select *Restart & Clear Output*.

# In this section we will dive deeper into Universal Dependencies, a framework that we already encountered in connection with syntactic parsing and morphological analysis in [Part II](../part_ii/03_basic_nlp.ipynb) and [Part III](01_multilingual_nlp.ipynb).
# 
# After reading through this section, you should:
# 
# - understand the goals of Universal Dependencies as a project
# - understand the Universal Dependencies as a framework for describing the structure of language
# - know the basics of the Universal Dependencies annotation schema
# - know how to explore Universal Dependencies annotations using spaCy

# ## A brief introduction to Universal Dependencies as a project
# 
# [Universal Dependencies](https://universaldependencies.org/introduction.html), or UD for short, is a collaborative project that has two overlapping aims:
#  1. to develop a common framework for describing the grammatical structure of diverse languages (de Marneffe et al. [2021](https://doi.org/10.1162/coli_a_00402))
#  2. to create annotated corpora – or treebanks – for diverse languages that apply this framework (Nivre et al. [2020](https://aclanthology.org/2020.lrec-1.497/))
#  
# In this way, the project seeks to enable the systematic description of syntactic structures and morphological features across various languages, which naturally also enables drawing comparisons between languages. 
# 
# The goal of achieving broad applicability across diverse languages lends the project the epithet "Universal", whereas the term "Dependencies" refers to the way the framework describes syntactic structures, which will be expanded on shortly below.
# 
# Linguistic corpora annotated for syntactic relations are often called *treebanks*, because syntactic structures are generally represented using tree structures. In this context, then, a treebank is simply a collection of syntactic trees, which have been consistently annotated using UD or some other framework.
# 
# The number of treebanks annotated using UD has grown steadily over the years (for a recent overview of 90 treebanks, see Nivre et al. [2020](https://aclanthology.org/2020.lrec-1.497/)). The design and creation of such treebanks has been documented in detail for various languages, such as Finnish (Haverinen et al. [2014](https://link.springer.com/article/10.1007/s10579-013-9244-1)), Wolof (Dione [2019](https://aclanthology.org/W19-8003/)) and Hindi/Urdu (Bhat et al. [2017](https://link.springer.com/chapter/10.1007/978-94-024-0881-2_24)).
# 
# To better understand the effort involved in UD as a project, one should acknowledge that developing a consistent annotation schema that can be used to describe the grammatical structure of diverse languages such as Finnish, Wolof and Hindi/Urdu is far from trivial. 
# 
# Additional challenges emerge from the intended use of UD treebanks: they are meant to serve both computational and linguistic research communities. As de Marneffe et al. ([2021: 302–303](https://doi.org/10.1162/coli_a_00402)) point out, the UD framework is a compromise between several competing criteria, which are provided in slightly abbreviated form below:
# 
#  1. UD needs to be reasonably satisfactory for linguistic analysis of individual languages
#  2. UD needs to be good for bringing out structural similarities across related languages
#  3. UD must be suitable for rapid, consistent annotation by a human annotator
#  4. UD must be easily comprehended and used by non-linguist users
#  5. UD must be suitable for computer parsing with high accuracy
#  6. UD must support well downstream natural language processing tasks
#  
# The need to balance these criteria is also reflected in the design of the UD framework, which is introduced in greater detail below.

# ## Basic assumptions behind Universal Dependencies

# The Universal Dependencies framework is strongly influenced by typologically-oriented theories of grammar. These theories seek to describe and classify languages based on their structural features (for an extensive discussion of the theoretical foundations of UD, see de Marneffe et al. [2021](https://doi.org/10.1162/coli_a_00402)).
# 
# The basic unit of analysis in UD is a word. The representation of syntactic relations, in turn, is based on **dependencies**, that is, relations that hold between words. In some aspects, however, UD diverges from traditional dependency grammars, mainly due to its need to serve the range of purposes described above (see Osborne & Gerdes [2019](https://doi.org/10.5334/gjgl.537)).
# 
# The description of linguistic structures in UD is based on three types of phrasal units: **nominals**, **clauses** and **modifiers** (de Marneffe et al. [2021: 257](https://doi.org/10.1162/coli_a_00402)). These phrasal units can consist of one or more words.
# 
# In this context, the notion of phrasal units refers to linguistic structures that are built around words that belong to particular word classes. Whereas nominals are used for representing things – often realised using nouns – clauses are used for representing events, which are built around verbs. Modifiers, in turn, rely on adjectives and adverbs to expand the meaning of both nominals and clauses.
# 
# The following sections examine each phrasal unit in greater detail.

# ### Nominals
# 
# Let's start by focusing on the first phrasal unit, nominals.
# 
# What UD defines as nominals have been described extensively in various linguistic theories, in which they have been treated, for example, as noun phrases (Keizer [2007](https://doi.org/10.1017/CBO9780511627699)) and nominal groups (Martin et al. [2021](https://doi.org/10.1080/00437956.2021.1957545)). What these definitions have in common is that nominals are generally built around nouns.
# 
# To explore nominals in UD, we will begin by importing three libraries: spaCy, Stanza and spacy-stanza, which were introduced in the [previous chapter](../part_iii/01_multilingual_nlp.ipynb).

# In[1]:


# Import the spaCy, Stanza and spacy-stanza libraries
import spacy
import stanza
import spacy_stanza


# We then use the `load_pipeline()` function from spacy-stanza to load a Stanza language model for English, which we store under the variable `nlp`.
# 
# We also pass the language code for English (`'en'`) to the argument `name` and the string `'tokenize, pos, lemma, depparse'` to the `processors` argument to load only the components that we need.
# 
# ```python
# # Load a Stanza language model for English into spaCy
# nlp_fi = spacy_stanza.load_pipeline(name='en', processors='tokenize, pos, lemma, depparse')
# ```
# 
# If you have not downloaded a language model for English yet, follow the instructions in the [previous section](01_multilingual_nlp.ipynb).

# In[2]:


# Download the default Stanza language model for English
stanza.download(lang='en', model_dir='../stanza_models')


# In[3]:


# Use the load_pipeline() function to load a Stanza model for English. Store the language 
# model under the variable 'nlp'. Only load the processors defined in the 'processors'
# argument.
nlp = spacy_stanza.load_pipeline(name='en', processors='tokenize, pos, lemma, depparse', dir='../stanza_models')


# In[4]:


# Call the variable to examine the language model
nlp


# This gives us a Stanza language model wrapped into a spaCy *Language* object!
# 
# If you wonder why we use a Stanza language model instead of a model native to spaCy, the reason is that the dependency parser in spaCy is not trained using a corpus annotated using UD.
# 
# In [Part II](../part_ii/03_basic_nlp.ipynb), we learned that spaCy language models for English are trained using the OntoNotes 5.0 corpus. This corpus uses a different schema for describing syntactic relations, which was originally developed for the Penn Treebank (PTB; Marcus et al. [1993](https://aclanthology.org/J93-2004/)). spaCy uses another tool to map the Penn Treebank relations to those defined in UD, but the relations defined in PTB only cover a subset of the relations defined in UD.
# 
# For this reason, we use the English language model from Stanza, which has been trained on [a corpus of texts](https://github.com/UniversalDependencies/UD_English-EWT) annotated using UD. 
# 
# We do, however, also want to use some capabilities provided by spaCy, such as the *displacy* module for visualising syntactic dependencies, as we learned in [Part II](../part_ii/03_basic_nlp.ipynb), which is why we use the Stanza language model via the spacy-stanza library.
# 
# Let's continue by importing the displacy module for visualising syntactic dependencies.

# In[5]:


# Import the displacy module from spaCy
from spacy import displacy


# We then define a string – "A large green bird" – that we feed to the language model under `nlp`, and assign the resulting *Doc* object under the variable `nominal`.

# In[6]:


# Feed a string to the language model; store the result under the variable 'nominal'
nominal = nlp('A large green bird')


# Next, we use the `render()` function to draw the syntactic dependencies between the *Tokens* in the *Doc* object `nominal_group`.
# 
# By passing the string `dep` to the argument `style`, we explicitly instruct *displacy* to visualise the syntactic dependencies (because *displacy* can also visualise [named entities](../part_ii/03_basic_nlp.ipynb).

# In[7]:


# Render the syntactic dependencies using the render() function from displacy
displacy.render(nominal, style='dep')


# This gives us a visualisation of the syntactic dependencies between the four *Tokens* that make up the *Doc* object `nominal`.
# 
# Three arcs lead out from the *Token* "bird" and point towards the *Tokens* "A", "large" and "green". This means that the noun "bird" acts as the **head**, whereas the three other *Tokens* are **dependents** of this head.
# 
# These dependencies are further specified by syntactic relations defined in the UD framework, which are given by the label below each arc. In this case, the head noun "bird" has two adjectival modifiers (`amod`), "large" and "green", and a determiner (`det`), "a".
# 
# If we loop over the *Tokens* in the *Doc* object under the variable `nominal` and print out the syntactic dependencies for each Token, which are available under the attribute `dep_`, we can see that the head noun "bird" has the dependency tag `root`.

# In[8]:


# Loop over each Token in the Doc object 'nominal'
for token in nominal:
    
    # Print out each Token and its dependency tag
    print(token, token.dep_)


# In other words, the entire syntactic structure of this nominal is built around a noun, which is then elaborated by modifiers, which will be discussed shortly below.
# 
# First, however, we turn our attention to another phrasal unit, namely clauses.

# ### Clauses
# 
# The clause plays a fundamental role in natural language. In *Introduction to Functional Grammar*, Halliday and Matthiessen ([2013](https://doi.org/10.4324/9780203431269): 10) observe that: 
# 
# > The clause is the central processing unit in the lexicogrammar — in the specific sense that it is in the clause that meanings of different kinds are mapped into an integrated grammatical structure.
# 
# These three distinct kinds of meanings – clause as a message, clause as an exchange and clause as a representation – are encoded into every clause. As messages, clauses have a thematic structure, which allows highlighting key information. As a form of exchange, clauses allow enacting social relationships, as they are used to give and demand information or things. Finally, as a form of representation, clauses enable representing all aspects of human experience: which entities participate in what kinds of processes, and under what kinds of circumstances (Halliday and Matthiessen [2013](https://doi.org/10.4324/9780203431269): 83–85).
# 
# To better understand what enables clauses to perform all these functions, let's consider their *rank* among different linguistic units, as defined by Halliday and Matthiessen ([2013](https://doi.org/10.4324/9780203431269): 9–10):
#  
#  1. clause
#  2. phrase / group
#  3. word
#  4. morpheme
# 
# The linguistic units at each rank consist of one or more units of the rank below. Clauses consist of phrases or groups (or nominals), which in turn consist of words that are made up of morphemes.
# 
# If we apply this idea to UD, we can think that clauses outrank nominals, which allows clauses to combine nominals into larger units (cf. de Marneffe et al. [2021: 258](https://doi.org/10.1162/coli_a_00402)).
# 
# To explore this idea further, let's define a string with the clause "I saw a large green bird" and provide the string as input to the language model under the variable `nlp`. We then store the result under the variable `clause`.
# 
# Just as above, we then use the `render()` function from the *displacy* module to visualise the syntactic dependencies.

# In[9]:


# Define a string object, feed it to the language model under 'nlp' and
# store the result under the variable 'clause'.
clause = nlp('I saw a large green bird.')

# Use displacy to render the syntactic dependencies
displacy.render(clause, style='dep')


# This gives us the syntactic relations that hold between the *Tokens* in the clause.
# 
# Before going any further, let's print out the dependency tags for each *Token*.

# In[10]:


# Loop over each Token in the Doc object 'clause'
for token in clause:
    
    # Print out each token and its dependency tag
    print(token, token.dep_)


# The output shows that the `root` or the head of the clause is the verb "saw". As the visualisation shows, two arcs lead out from the `root` towards the *Tokens* "I" and "bird".
# 
# Both "I" and "a large green bird" are nominals and dependents of the verb "saw", which is the head of the clause.  The pronoun "I" acts as the nominal subject of the clause, as identified by the label (`nsubj`), whereas the nominal "a large green bird" is the object (`obj`).
# 
# Note that syntactic dependencies are always drawn between heads: the arcs lead out from the head verb of the clause and terminate at the heads of the nominals. These heads may then have their own dependencies, as illustrated by the nominal "a large green bird", which was discussed below.
# 
# Just like nominals, clauses can be expanded into larger units, as exemplified below.

# In[11]:


# Define another example and feed it to the language model under 'nlp'
clause_2 = nlp('I saw a large green bird outside and headed out immediately.')

# Use displacy to render the syntactic dependencies
displacy.render(clause_2, style='dep')


# This adds another arc, which leads from the head verb "saw" to the verb "headed", and has the relation `conj` or conjunct. The conjunct is used to join together two clauses: 
# 
#  1. I saw a large green bird outside 
#  2. and headed out immediately.
#  
# This illustrates how clauses can also be expanded into larger units that consist of multiple clauses. In this case, the clauses participating in a larger unit may be identified by the dependency relation `conj` drawn between verbs.
# 
# Note, however, that the `conj` relation can also be used within nominals to join nouns together, as exemplified below.

# In[12]:


# Define another example and use displacy to render the syntactic dependencies
displacy.render(nlp('cats, dogs and birds'), style='dep')


# This illustrates how UD uses the same relation to describe syntactic relations between words in different phrasal units.
# 
# For this reason, one needs to pay attention to both part-of-speech tags *and* syntactic dependencies when querying UD annotations to identify the phrasal unit in question.
# 
# To exemplify, when the relation `conj` is drawn between nouns, one may assume that the phrasal unit is a nominal. Alternatively, if the `conj` relation exists between verbs, the unit in question is a clause.

# ### Modifiers
# 
# The final type of phrasal unit to be discussed are modifiers, which allow refining the meaning of both clauses and nominals.
# 
# Let's start with a simple example of modifiers in a nominal.

# In[13]:


# Define a string object, feed it to the language model under 'nlp' and
# store the result under the variable 'mod_example'.
modifier_n = nlp('A very nasty comment')

# Use displacy to render the syntactic dependencies
displacy.render(modifier_n, style='dep')


# The arc that leads from the head noun "comment" to the adjective "nasty" has the relation `amod`, which stands for adjectival modifier.
# 
# In addition, the adjective "nasty" has further dependent, the adverb "very", which acts as its adverbial modifier (`advmod`).
# 
# Both of these modifiers serve to refine the meaning of the head noun "comment".
# 
# Just as we saw with the conjunct relation (`conj`) above, these relations can be applied to both clauses and nominals.
# 
# Consider the example below, in which the adverb "slowly" is a dependent of the head verb "opened".

# In[14]:


# Define a string object, feed it to the language model under 'nlp' and
# store the result under the variable 'mod_example'.
modifier_c = nlp('The door opened slowly.')

# Use displacy to render the syntactic dependencies
displacy.render(modifier_c, style='dep')


# Clauses can also be modified by clauses, as shown by the adverbial clause modifier (`advcl`) in the example below.

# In[15]:


# Define a string object, feed it to the language model under 'nlp' and
# store the result under the variable 'mod_example'.
modifier_c2 = nlp('The door opened slowly, without making a sound.')

# Use displacy to render the syntactic dependencies
displacy.render(modifier_c2, style='dep')


# ## Understanding the annotation schema
# 
# So far, we have mainly discussed the description of syntactic relations within the Universal Dependencies (UD) framework.
# 
# In addition to the [37 syntactic relations](https://universaldependencies.org/u/dep/) defined in UD, the framework provides [a rich schema for describing morphology](https://universaldependencies.org/u/overview/morphology.html), that is, the *form* of words.
# 
# The UD schema for morphology contains three levels of representation:
#  1. A lemma, or the base form of the word
#  2. A part-of-speech tag determining the word class to which the word belongs
#  3. A set of features that define the lexical and grammatical features of the word
#  
# UD defines [17 part-of-speech or word classes](https://universaldependencies.org/u/pos/index.html), which can be divided into three groups:
# 
#  1. Open class or lexical words: `ADJ ADV INTJ NOUN PROPN VERB`
#  2. Closed class or grammatical words: `ADP, AUX, CCONJ, DET, NUM, PART, PRON, SCONJ`
#  3. Other: `PUNCT, SYM, X`
#  
# UD also defines a large number of [lexical and inflectional features](https://universaldependencies.org/u/feat/index.html) for describing morphological features, that is, the word forms.
# 
# UD defines morphological features using two components, *names* and *values*. As we learned in [Part II](../part_ii/03_basic_nlp.ipynb), spaCy stores morphological features under the `morph` attribute of a *Token* object.
# 
# Let's define a quick example, feed it to the language model under the variable `nlp` and print out each *Token* and its morphological features.  

# In[16]:


# Define an example sentence; feed it to the language
# model and store the result under the variable 'books'
books = nlp('I like those books.')

# Loop over each Token in the Doc object
for token in books:
    
    # Print out each Token, its part-of-speech tag and
    # morphological features. Separate these using strings
    # that contain tabulator '\t' characters for cleaner
    # output.
    print(token, '\t', token.pos_, '\t', token.morph)


# In the result, each *Token* on the left-hand side is accompanied by its part-of-speech tag and morphological features on the right.
# 
# Note how the morphological features differ according to the part-of-speech tag of the *Token*.
# 
# For the pronoun (`PRON`) "I", the language model predicts four types of morphological features: `Case`, `Number`, `Person` and `PronType` (pronoun type). Verbs such as "like", in turn, are assigned features for `Mood`, `Tense` and `VerbForm`.
# 
# As we will learn later in [Part III](03_pattern_matching.ipynb), morphological features can be used to perform very specific queries for particular linguistic structures.

# ## Exploring parse trees using spaCy
# 
# spaCy offers powerful means exploring syntactic dependencies through the *Token* object.
# 
# Let's start by defining another example, feeding it to the language model and visualising its syntactic dependencies.

# In[17]:


# Define an example string and feed it to the language model under 'nlp',
# store the result under the variable 'tree'.
tree = nlp('I never saw the bird, because it had flown away.')

# Use displacy to render the syntactic dependencies
displacy.render(tree, style='dep')


# In this example, the head verb "saw" has a dependent, "flown", which are connected by the dependency relation `advcl` – an adverbial clause modifier.
# 
# If we wish to retrieve everything that modifies the main clause "I never saw the bird", we can use the `subtree` attribute of a *Token* object.
# 
# Let's explore this by looping over each *Token* in the *Doc* object `tree`.
# 
# If the *Token* has the dependency `advcl` under the `dep_` attribute, we print out the *Token*, its index in the *Doc* object and whatever is stored under the `subtree` attribute.

# In[18]:


# Loop over each Token in the Doc object 'tree'
for token in tree:
    
    # Check if the Token has the dependency relation 'acl:relcl',
    # which stands for a relative clause
    if token.dep_ == 'advcl':
        
        # If the Token has this dependency, use the subtree attribute
        # to fetch all dependents below this Token. The subtree attribute
        # returns a generator, so cast the result into a list and print.
        print(token, token.i, list(token.subtree))


# If you compare this output to the syntactic dependencies visualised above, you should see that the `subtree` attribute returns every dependent of the *Token*, and the *Token* itself.
# 
# If we want to retrieve only the *Tokens* that depend on the current *Token*, we can use the `children` attribute.
# 
# Let's use the index of the *Token* to retrieve its children and print out the result.

# In[19]:


# Retrieve Token at index 9 in the Doc object and fetch its children.
# This returns a generator, so cast the output into a list before printing.
print(list(tree[9].children))


# As you can see, the `children` attribute does not return the *Token* itself, but only includes the dependents.
# 
# spaCy also allows retrieving the immediate dependents of a *Token* using the attributes `lefts` and `rights`.

# In[20]:


# Retrieve the syntactic dependents left and right of the Token at
# index 9 in the Doc object 'tree'. Cast the results into lists and
# print.
print(list(tree[9].lefts), list(tree[9].rights))


# We can also move the other way – up the parse tree – using the `head` and `ancestors` attributes.
# 
# Let's start by examining the auxiliary verb "have" immediately left of the verb "flown" at index 8 of the *Doc* object.

# In[21]:


# Retrieve the Token
tree[8]


# To retrieve the *Token* that acts as the head of the auxiliary verb, we can use the `head` attribute, as we learned in [Part II](../part_ii/03_basic_nlp.ipynb#Syntactic-parsing).
# 
# Let's retrieve the head for the auxiliary verb "had" at index 8 of the *Doc* object `tree`.

# In[22]:


# Retrieve the 'head' attribute of the Token
tree[8].head


# This, however, only gives us the immediate head, that is, the verb "flown".
# 
# To retrieve the head and all its heads, we can use the `ancestors` attribute. This attribute returns a generator object, which must be cast into a list for examination.

# In[23]:


# Retrieve the ancestors for Token at index 8. Cast the result into a list.
list(tree[8].ancestors)


# You can think of this attribute as tracing a way back through the dependencies all the way to the root of the parse tree.
# 
# Let's loop over the ancestors and print out each *Token* together with its index, head and syntactic dependency.

# In[24]:


# Loop over each Token in the list of ancestors for Token at index 8.
for token in list(tree[8].ancestors):
    
    # Print out each Token, its index, head and dependency.
    print(token, token.i, token.head, token.dep_)


# As you can see, the first head of the auxiliary verb "had" is the verb "flown" at index 9 of the *Doc* object, which in turn is a dependent of the verb "saw" at index 2. The verb "saw" is also the root of the parse tree.

# ## A final word on evaluation
# 
# Language models trained on Universal Dependencies treebanks are generally accompanied by information on how well the models can predict the linguistic features defined in the UD annotation schema.
# 
# As we learned in [Part II](../part_ii/05_evaluating_nlp.ipynb), the performance of models is evaluated against human-annotated data, a so-called gold standard or ground truth.
# 
# For dependency parsing, how well a model performs is traditionally measured using two metrics:
# 
#  - UAS, or *unlabeled attachment score*, is simply the percentage of words that are assigned the correct head. 
#  - LAS, or *labeled attachment score*, is the percentage of words that are assigned the correct head *and* the correct dependency tag (or "label").
#  
# Let's define a quick example to examine these metrics.

# In[25]:


# Define an example string and feed it to the language model under 'nlp',
# store the result under the variable 'las'.
las = nlp('I went to the cinema.')

# Use displacy to render the syntactic dependencies. Set the 'collapse_punct'
# argument to False.
displacy.render(las, style='dep', options={'collapse_punct': False})


# If the parse tree above were annotated by a human, we could then feed the same text to a language model and compare the model output to the human-annotated parse tree.
# 
# To calculate the unlabeled attachment score (UAS), we would simply calculate how many words were assigned the correct head by the model.
# 
# When calculating the labeled attachment score (LAS), a prediction is only considered correct if the model assigns the correct head to the word *and* correctly predicts the syntactic relation between these words.
# 
# However, UAS and LAS become problematic when considering the cross-linguistic goals of UD as a project: one should also be able to compare the performance of models *across languages*.
# 
# Consider, for instance, the Finnish equivalent of the example above: "*Menin elokuviin.*" ("I went to the cinema.").
# 
# If the English language model were to predict the wrong head for a single word, the model would nevertheless achieve a UAS or LAS score of $5/6 \approx 0.83$.
# 
# If the Finnish parser, in turn, would make a single mistake, the corresponding score would be $2/3 \approx 0.66$.
# 
# Zeman et al. [2018: 5](https://aclanthology.org/K18-2001.pdf) summarise the problem succinctly:
# 
# > ... function words often correspond to morphological features in other languages. Furthermore, languages with many function words (e.g. English) have longer sentences than morphologically rich languages (e.g. Finnish), hence a single error in Finnish costs the parser significantly more than an error in English according to LAS.
# 
# For this reason, several alternative metrics have been proposed for measuring the performance of language models for dependency parsing.
# 
# CLAS, or *content-labeled attachment score*, is like LAS, but ignores function words (e.g. words with relations `aux` `case` `cc` `clf` `cop` `det` `mark`) and punctuation (`punct`) when calculating the score. Only content words are counted (Nivre & Fang [2017](https://aclanthology.org/W17-0411.pdf)).
# 
# MLAS, or *morphologically-aware labeled attachment score*, is largely similar to CLAS, but also evaluates whether the part-of-speech tag and selected morphological features have been predicted correctly (Zeman et al. [2018: 5](https://aclanthology.org/K18-2001.pdf))
# 
# BLEX, or *bilexical dependency score*, is like MLAS, but morphological information is replaced by lemmas (Zeman et al. [2018: 5](https://aclanthology.org/K18-2001.pdf)).

# This section should have given you an idea of Universal Dependencies as a project and an annotation schema.
# 
# In the [next section](03_pattern_matching.ipynb), we will learn how to search linguistic annotations for patterns.
