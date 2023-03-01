#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import openai


openai.api_key = "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"

response = openai.Completion.create(
  model="code-davinci-002",
  prompt="create a fab function that output 100， use rec",
  temperature=0,
  max_tokens=300,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response)
