import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain_community.llms.ctransformers import LLM


def getLLamaresponse(input_text, no_words, blog_style):
    llm = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML', model_type='llama', config={'max_new_tokens': 256, 'temperature': 0.01})

    template = """
    write a blog for {blog_style} job profile for a topic {input_text} with {no_words} words."""
    #prompt
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template=template)
    
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    print(response)
    return response

st.set_page_config(page_title="Generate Blogs",
                     page_icon="📝",
                     layout="centered",
                        initial_sidebar_state="collapsed")


st.header("Generate Blogs ")

input_text = st.text_input("Enter the text you want to generate a blog on")

col1, col2 = st.columns([5,5])
with col1:
    no_words = st.text_input("Enter the number of words you want in the blog")
with col2:
    blog_style = st.selectbox("Writing the blog for", ('Researchers', 'Students', 'Professionals', 'General Audience'), index=0)
    
    
submit = st.button("Generate Blog")

if submit:
    st.write(getLLamaresponse(input_text, no_words, blog_style))