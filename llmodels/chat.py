#!/usr/bin/env python3
# -*- coding:utf-8 -*- 
import click
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from pathlib import Path


Notice = "USER >"
@click.command(name="chat")
@click.option("-m", "--model_name_or_path")
def chat_command(model_name_or_path: str):

    click.secho("[*] Loading your model...", fg="blue", bold=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    except:
        click.secho(
                f"[-] The model_name_or_path you have provided {model_name_or_path} is not valid",
                fg="red",
                bold=True
        )
        return 

    click.secho(
        "[+] Model loaded successfully. Happy chatting!\n\n", fg="green", bold=True
    )

    while True:
        user_input = input(Notice)
        inputs = tokenizer(user_input.replace(Notice, ""), return_tensors="pt")
        outputs = model.generate(**inputs, num_beams=5, max_new_tokens=50, early_stopping=True, no_repeat_ngram_size=2)
        text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("MODEL> {}".format(text_output[0]))

if __name__ == "__main__":
    chat_command()