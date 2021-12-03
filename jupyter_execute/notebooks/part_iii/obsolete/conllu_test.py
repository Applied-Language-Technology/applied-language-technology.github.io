#!/usr/bin/env python
# coding: utf-8

# In[1]:


import conllu


# In[2]:


data = open("data/GUM_whow_parachute.conllu", "r", encoding="utf-8")


# In[3]:


sentences = list(conllu.parse_incr(data))


# Check the type of objects.

# In[4]:


type(sentences[0])


# The discourse segment ID can be retrieved from the `metadata` attribute of a *TokenList* object.

# In[5]:


sentences[0].metadata


# The discourse annotations are always contained under the key `misc`.

# In[6]:


print(sentences[0].serialize())


# Parse the *TokenList* objects into spaCy *Docs*: https://spacy.io/api/doc
# 
# Then register custom attributes for RST and sentence type.

# In[7]:


from spacy.training.converters import conllu_to_docs
from spacy.tokens.span_group import SpanGroup
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
import spacy

nlp = spacy.load('en_core_web_sm')


# Anticipating the need to store discourse annotations into spaCy *Doc* and *Span* objects, we register several custom attributes with spaCy as instructed in [Part II](../part_ii/04_basic_nlp_continued.html#adding-custom-attributes-to-spacy-objects).

# In[8]:


# Register custom attributes named 'sentence_type' and 'sentence_id' 
# for Doc object; set default value to None for both
Doc.set_extension('sentence_type', default=None)
Doc.set_extension('sentence_id', default=None)

# Register custom attributes named 'edu_id'
Span.set_extension('edu_id', default=None)
Span.set_extension('target_id', default=None)
Span.set_extension('relation', default=None)


# In[9]:


def tokens_to_doc(nlp_object, token_list):
    
    # Assert the input is of correct type, e.g. a TokenList
    assert type(token_list) == conllu.models.TokenList
    
    # Collect the form of tokens from the TokenList object
    words = [token['form'] for token in token_list]
    
    # Collect information on whether token should be followed by empty space
    spaces = [False if token['misc'] is not None 
              and 'SpaceAfter' in token['misc'] 
              and token['misc']['SpaceAfter'] == "No"
              else True for token in token_list]
    
    # Create a spaCy Doc object manually
    doc = Doc(vocab=nlp.vocab, words=words, spaces=spaces)
    
    return doc


# In[10]:


def convert_conllu_to_doc(sentence):
    
    # Assert the input is of correct type, e.g. a TokenList
    assert type(sentence) == conllu.models.TokenList
    
    # Next, we convert the TokenList object into a spaCy Doc object
    doc = tokens_to_doc(nlp, sentence)
    
    # Pass the Doc object through the pipeline
    for name, processor in nlp.pipeline:
        
        doc = processor(doc)
    
    # Next, we extract the sentence ID and type from the conllu TokenList
    # object and assign them to variables 'sent_id' and 'sent_type'.
    sent_id = sentence.metadata['sent_id']
    sent_type = sentence.metadata['s_type']
        
    # We then create placeholder list for holding Token indices that mark
    # the boundaries of elementary discourse units, their identifiers and 
    # discourse relations
    edus = []
    edu_ids = []
    target_ids = []
    disc_rels = []
    
    # Next, we loop over each item (Token) in the TokenList object 'sentence'
    # to examine its attributes.
    for token in sentence:
                        
        # Check if the 'misc' dictionary under Token exists and contains the 
        # key 'Discourse'.
        if token['misc'] is not None and 'Discourse' in token['misc']:
            
            # If both conditions are true, append token index to the 'discourse_units'
            # list that holds the indices of discourse unit boundaries.
            edus.append(sentence.index(token))
            
            # Get the discourse relation definition stored under the key 'Discourse'
            rel_definition = token['misc']['Discourse']
            
            # The relation definitions are provided as strings with the following pattern:
            #
            #  preparation:1->11
            #
            # The relation name is on the left-hand side of a colon, whereas the 
            # right-hand side contains identifiers for the participating units. 
            # The first identifier is always the identifier for the current unit. 
            # The second is the "target" of the relation. To separate them, we 
            # use the split() method to split at colon.
            rel, rel_ids = rel_definition.split(':')
            
            # We then check if the 'rel_ids' string contains '->' indicating a 
            # relation. The root element does not have this.
            if '->' in rel_ids:
                
                # Get the unit and target identifiers by splitting at '->'
                edu_id, target_edu_id = rel_ids.split('->')
                
                # Add EDU and target EDU ids to list
                edu_ids.append(edu_id)
                target_ids.append(target_edu_id)
            
            # Define alternative steps for processing root elements
            else:
                
                # Add root element identifier to lists
                edu_ids.append(rel_ids)
                
                # Add target EDU id to list as None
                target_ids.append(None)
                
            # Add discourse relation to list
            disc_rels.append(rel)
            
    # Finally, append the length of the TokenList object to the 'discourse_units' list 
    # to mark the boundary of the final discourse unit!
    edus.append(len(sentence))
    
    # Next, we create a placeholder list to hold slices of the Doc object that correspond
    # to discourse units.
    edu_spans = []

    # Proceed to loop over the discourse units. To do so, we use Python's range() function, 
    # to count from zero to the length of the 'edus' list minus one. This is  because the 
    # final item will never mark the beginning of a discourse unit. We use these numbers 
    # as indices for list items during the following loop!
    for edu in range(len(edus) - 1):
        
        # Get the start of the discourse unit by fetching the current item in the list
        start = edus[edu]
        
        # Get the end of the discourse unit by fetching the next item in the list
        end = edus[edu + 1]
        
        # The 'start' and 'end' variables now hold Token indices; use them to slice the
        # spaCy Doc object 'doc' into Span objects. Add Spans to list 'edu_spans'.
        edu_spans.append(doc[start:end])
        
    # This is a good point to check that the number of EDUs matches the number of identifiers
    # and relations. This will throw an error if something goes wrong.
    assert len(edu_spans) == len(edu_ids) == len(target_ids) == len(disc_rels)
    
    # Next we create a spaCy SpanGroup object that holds all our Spans for elementary discourse
    # units. The 'doc' argument takes the Doc object that the Spans belong to as input, whereas
    # 'name' defines the key used to retrieve the spans from the 'spans' attribute of the Doc
    # object. Finally, 'spans' takes a list of Span objects to be included in the Span group.
    span_group = SpanGroup(doc=doc, name="edus", spans=edu_spans)
    
    # Assign the SpanGroup to the 'spans' attribute of the Doc object under the key 'edus'
    doc.spans['edus'] = span_group
    
    for i, span in enumerate(doc.spans['edus']):
        
        span._.edu_id = edu_ids[i]
        span._.target_id = target_ids[i]
        span._.relation = disc_rels[i]
        
    return doc


# In[11]:


discourse = [convert_conllu_to_doc(sentence) for sentence in sentences]


# In[12]:


docs = Doc.from_docs(discourse)


# In[13]:


docs


# In[14]:


docs.spans['edus']


# In[15]:


data = open("GUM_whow_parachute.conllu", "r", encoding="utf-8")

conllu = data.read()

docs = list(conllu_to_docs(conllu, no_print=True))


# New plan
# 
# - use `conllu` to extract metadata, then use `TokenList.serialize()` and feed this to spaCy `conllu_to_docs` to create a Doc object.

# In[16]:


# Convert the TokenList into a CoNLL-U compliant string object using
# the serialize() method. Then feed this string to the conllu_to_docs()
# function from spaCy. Setting the no_print argument to True prevents 
# any output at this stage.
# doc = list(conllu_to_docs(sentence.serialize(), no_print=True))

# Because the conllu_to_docs() function returns a Python generator object, we
# must cast the output into a list to examine its contents. We then access 
# the first item at index 0 to access the Doc object and update the variable.
# doc = doc[0]


# In[17]:


doc_1 = nlp("This is a test.")


# In[18]:


doc_1[0].head


# In[ ]:




