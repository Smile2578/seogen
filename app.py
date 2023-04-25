import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ['OPENAI_API_KEY'] = apikey

st.title('Nunny SEO Generator')

prompt = st.text_input('Ecrit ton prompt ici')

title_template = PromptTemplate(
   input_variables = ['topic'],
   template = 'ecrit moi un titre h1 pour un article de blog sur {topic} en respectant les standards SEO. '
)

script_template = PromptTemplate(
   input_variables = ['title', 'wikipedia_research'],
   template = 'ecrit moi un plan chapitré pour un article de blog, en respectant les standards SEO, basé sur ce titre: {title} , tout en s\'inspirant de ce que tu peux trouver sur wikipedia: {wikipedia_research}'
)

content_template = PromptTemplate(
    input_variables = ['script'],
    template = 'ecrit moi le contenu avec les titres des chapitres en h2 et sous-chapitres en h3 de l\'article de blog, en respectant les standards SEO, basé sur ce plan: {script}'
)

#Mémoire


title_memory = ConversationBufferMemory(
    input_key='topic',
    memory_key='chat_history'
)
script_memory = ConversationBufferMemory(
    input_key='title',
    memory_key='chat_history'
)
content_memory = ConversationBufferMemory(
    input_key='script',
    memory_key='chat_history'
)





#LLMS

llm = OpenAI(temperature=0.9, max_tokens=3600, top_p=1, frequency_penalty=0, presence_penalty=0.3)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
content_chain = LLMChain(llm=llm, prompt=content_template, verbose=True, output_key='content', memory=content_memory)

wiki = WikipediaAPIWrapper()

# Show the prompt

if prompt: 
   title = title_chain(topic=prompt)
   wiki_research = wiki.run(prompt)
   script = script_chain(title=title, wikipedia_research=wiki_research)
   content = content_chain(script=script)
   st.write(title)
   st.write(content)

   with st.expander('Historique des Titres'):
      st.info(title_memory.buffer)

   with st.expander('Historique des plans'):
      st.info(script_memory.buffer)

   with st.expander('Recherche Wiki'):
        st.info(wiki_research)

   with st.expander('Historique du contenu du plan'):
      st.info(content_memory.buffer)