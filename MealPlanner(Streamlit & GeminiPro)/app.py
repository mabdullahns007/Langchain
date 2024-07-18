#@title Setting up the Auth
import os
import google.generativeai as genai
from google.colab import userdata

from IPython.display import display
from IPython.display import Markdown


genai.configure(api_key="")

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True,
                             temperature=0.7)

from langchain.prompts import PromptTemplate

# meal template
meal_template = PromptTemplate(
    input_variables=["ingredients"],
    template="Give me an example of 2 meals that could be made using the following ingredients: {ingredients}",
)

# gangster template
gangster_template = """Re-write the meals given below in the style of a New York mafia gangster:

Meals:
{meals}
"""

gangster_template = PromptTemplate(
    input_variables=['meals'],
    template=gangster_template
)

from langchain.chains import LLMChain
from langchain.chains import SequentialChain

meal_chain = LLMChain(
    llm=llm,
    prompt=meal_template,
    output_key="meals",  # the output from this chain will be called 'meals'
    verbose=True
)

gangster_chain = LLMChain(
    llm=llm,
    prompt=gangster_template,
    output_key="gangster_meals",  # the output from this chain will be called 'gangster_meals'
    verbose=True
)

overall_chain = SequentialChain(
    chains=[meal_chain, gangster_chain],
    input_variables=["ingredients"],
    output_variables=["meals", "gangster_meals"],
    verbose=True
)

import streamlit as st
st.title("Meal planner")
user_prompt = st.text_input("A comma-separated list of ingredients")

if st.button("Generate") and user_prompt:
    with st.spinner("Generating..."):
        output = overall_chain({'ingredients': user_prompt})

        col1, col2 = st.columns(2)
        col1.write(output['meals'])
        col2.write(output['gangster_meals'])
