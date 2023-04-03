#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


tokenizer = AutoTokenizer.from_pretrained("../Cerebras-GPT-111M")
model = AutoModelForCausalLM.from_pretrained("../Cerebras-GPT-111M")

text = "Generative AI is"

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
generated_text = pipe(text, max_length=50, do_sample=False, no_repeat_ngram_size=2)[0]
print(generated_text['generated_text'])

