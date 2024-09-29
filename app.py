import streamlit as st
from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate

## Function To get response from LLaMa 2 model
def getLLamaresponse(input_text, no_words, blog_style):
    ### LLaMa2 model
    llm = CTransformers(
        model='/Users/ridhunks/Documents/models/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={'max_new_tokens': 256, 'temperature': 0.01}
    )
    
    ## Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    
    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )
    
    ## Generate the response from the LLaMa 2 model
    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    response = llm.invoke(formatted_prompt)  # Use `invoke` instead of `__call__`
    
    print(response)
    return response

# Streamlit app configuration
st.set_page_config(
    page_title="Generate Blogs",
    page_icon='🤖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

# Display response
if submit:
    response = getLLamaresponse(input_text, no_words, blog_style)
    st.write(response)