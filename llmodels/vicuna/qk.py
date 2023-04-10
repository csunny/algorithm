#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import pipeline

BASE_MODEL = "../vicuna-13b"
model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    offload_folder="./data/vicuna",
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

text = "Generative AI is"
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
generated_text = pipe(text, max_length=50, do_sample=False, no_repeat_ngram_size=2)[0]
print(generated_text['generated_text'])


