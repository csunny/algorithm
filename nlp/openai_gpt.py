#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import openai

openai.api_key = ""

def text_2_code(text):
    response = openai.Completion.create(
    model="code-davinci-002",
    prompt=text,
    temperature=0,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response


def text_2_sql(text):

    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=text,
        temperature=0,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["#", ";"]
    )
    return response

def text_2_image(text):
    response = openai.Image.create(
        prompt=text,
        n = 1,
        size="1024x1024"
    )
    return response

if __name__ == "__main__":
    # text = "### Postgres SQL tables, with their properties:\n#\n# Employee(id, name, department_id)\n# Department(id, name, address)\n# Salary_Payments(id, employee_id, amount, date)\n#\nA query to list the names of the departments which employed more than 10 employees in the last 3 months" 
    # res = text_2_sql(text)
    # print(res)

    text = "a a beautiful girl"
    res = text_2_image(text)
    print(res["data"][0]["url"])