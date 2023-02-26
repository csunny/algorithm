#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import io

import warnings
from PIL import Image
from stability_sdk import client

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
# Sign up for an account at the following link to get an API Key.
# https://beta.dreamstudio.ai/membership


os.environ['STABILITY_KEY'] = 'sk-Mmq69dSTDwats3aN1b2Rnu9fx8YtwqKQmbfY2uXn4IxlMTrG'

# Set up our connection to the API
stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine='stable-diffusion-v1-5',
)


# Set up our initial generation parameters
answers = stability_api.generate(
        prompt="generate a man",
        seed = 992446758,
        steps = 30,
        cfg_scale=8.0,
        width=512,
        height=512,
        samples=1,
        sampler=generation.SAMPLER_K_DPMPP_2M
)

for resp in answers:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            img = Image.open(io.BytesIO(artifact.binary))
            print(img)
            img.save(str(artifact.seed)+ ".png") #

