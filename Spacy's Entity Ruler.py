#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy


# Here we have taken a random text "Gopaldi was referenced in Mr.Deeds.". Where Gopaldi is a GPE and Mr.Deeds is a FILM entity. So now we will see correctly spcay can indentify these entities in the following lines.

# In[6]:


nlp=spacy.load("en_core_web_sm")
text="Gopaldi was referenced in Mr.Deeds."


# In[7]:


doc=nlp(text)


# In[8]:


for ent in doc.ents:
    print(ent.text,ent.label_)


# In[9]:


ruler=nlp.add_pipe("entity_ruler")


# In[12]:


nlp.analyze_pipes()


# In[15]:


patterns = [
    {"label":"GPE","pattern":"Gopaldi"}
]


# In[16]:


ruler.add_patterns(patterns)


# In[18]:


doc2=nlp(text)
for ent in doc2.ents:
    print(ent.text,ent.label_)


# The problem here is - still "Gopaldi" is a person entity even after adding labels as GPE. It's because, we can see the the pipeline analyzer that "ner" comes before "entity_ruler". As per "ner" Gopaldi is a person and we added the label in entity_ruler. thus it didn't work. So what can we do, is to move up the "entity_ruler" before"ner", so that our new labeling system works properly. 

# In[24]:


#let's load the en_core_web_sm model again.
nlp2=spacy.load("en_core_web_sm")


# In[25]:


#moving entity_ruler before ner in the pipeline.
ruler=nlp2.add_pipe("entity_ruler",before="ner")


# In[26]:


ruler.add_patterns(patterns)


# In[27]:


doc=nlp2(text)


# In[28]:


for ent in doc.ents:
    print(ent.text,ent.label_)


# In[30]:


# now let's analyze our new modified pipeline.
nlp2.analyze_pipes()


# Now let's solve the issue for Mr.Deeds as well. 

# In[31]:


nlp3=spacy.load("en_core_web_sm")


# In[32]:


ruler=nlp3.add_pipe("entity_ruler",before="ner")


# In[33]:


patterns=[
    {"label":"GPE","pattern":"Gopaldi"},
    {"label":"FILM","pattern":"Mr.Deeds"}
]


# In[34]:


ruler.add_patterns(patterns)


# In[35]:


doc=nlp3(text)


# In[36]:


for ent in doc.ents:
    print(ent.text,ent.label_)


# In[ ]:




