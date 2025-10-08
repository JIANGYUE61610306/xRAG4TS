import os
import openai
import sys
import json
import numpy as np
import pandas as pd 
import random
# import re
from tqdm import tqdm
import time
import re

seed='2'

#### OpenAI
import openai
from openai import OpenAI



# These are the basic functions to call ChatGPT
def before_retry_fn(retry_state):
    if retry_state.attempt_number > 1:
        print(f"Retrying API call. Attempt #{retry_state.attempt_number}, f{retry_state}")


# @retry(wait=wait_fixed(5)+ wait_random(0, 5),stop=stop_after_attempt(6), before=before_retry_fn)
def completion_with_backoff(api_type="chat", **kwargs):
    key_list = ["YourKEY"]  ### Replace your openAI key here please.
    client = OpenAI(api_key=random.choice(key_list))

    if api_type == "chat":
        return client.chat.completions.create(**kwargs)
    elif api_type == "responses":
        return client.responses.create(**kwargs)
    else:
        raise ValueError(f"Unknown api_type: {api_type}")


def get_completion(prompt, gpt_model="gpt3.5", max_tokens=128, seed=None):
    """
    gpt_model 可选:
      - "gpt3.5"  → gpt-3.5-turbo-0125
      - "gpt4"    → gpt-4
      - "o1" / "o1-mini" / "o1-pro" → 用 Responses API
    """
    if gpt_model in ("gpt3.5", "gpt-3.5-turbo"):
        model = "gpt-3.5-turbo-0125"
        response = completion_with_backoff(
            api_type="chat",
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            seed=seed
        )
        return response.choices[0].message.content.strip()

    elif gpt_model in ("gpt4", "gpt-4"):
        model = "gpt-4"
        response = completion_with_backoff(
            api_type="chat",
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
            seed=seed
        )
        return response.choices[0].message.content.strip()

    elif gpt_model in ("o1", "o1-mini", "o1-pro"):
        print('Now, using o1 as your reasoner!')
        model = gpt_model
        response = completion_with_backoff(
            api_type="responses",
            model=model,
            input=prompt,
            max_output_tokens=max_tokens,
            reasoning={"effort": "medium"}  
        )
        return response.output_text.strip()

    else:
        raise ValueError(f"Unknown gpt_model: {gpt_model}")


target_parking_data = pd.read_hdf('./data/nottingham11.h5')
source_parking_data = pd.read_hdf('./data/carpark_05_06_15min.h5')
true = np.load('ytrue_train_transfer_Nottingham.npy')
pred = np.load('ypred_train_transfer_Nottingham.npy')

target_carpark_des_list = []
file_path = "carpark_des_list_nottingham.txt"

with open(file_path, "r") as file:

    for line in file:

        line = line.strip()

        if line:
            try:

                target_carpark_des_list.append(line)
            except ValueError:

                print("Warning: Skipped line as it cannot be converted to an integer:", line)


source_carpark_des_list = []
file_path = "carpark_des_list_nottingham.txt"

with open(file_path, "r") as file:

    for line in file:

        line = line.strip()

        if line:
            try:

                source_carpark_des_list.append(line)
            except ValueError:

                print("Warning: Skipped line as it cannot be converted to an integer:", line)

retrieve_results = np.loadtxt("retrieve_results.csv", delimiter=",", skiprows=1, dtype=int)


instruction_text = '<s>[INST] Role: You are an AI agent specialized in cross-city transferlearning analysis for parking availability. Objective: Do not outputfuture parking values. Instead, use reasoning and causal inference to extract useful, generalizable hints that relate the inputs (textual context, source long-term sequence, target short-term history, and simulation predictions) to the ground truth sequence. These hints will be used later to fine-tune a student model (e.g., LLaMA-3.1-8B). Focus on: (i) bias/scale calibration of simulation predictions, (ii) lag/seasonality/day-of-week effects, (iii) cross-city pattern alignment. Input Data: (1) Source city textual information: {}. (2) Source city long-term sequenc:Retrieved multi-week availability series with high semantic similarity to the target carpark {}. (3) Prediction horizon: given 3 hours historical input from {} to {} on {}, and predict future parking lots from {} to {}. (4) Historical Records: Parking data for the past 3 hours {}. (5) Simulation predictions: Forecasts for parking availability for the next 3 hours based on simulation models {}.(6)Target city textual information:{}. Training-time supervision: The ground truth for the horizon is provided as {}. Use it only to infer relationships (e.g., bias/scale, residual directions, regime shifts). Do not echo the ground truth or produce numeric forecasts. Analysis Goals (what to extract): a. Cross-city alignment: Which parts of the source long-term sequence best align with the target history (e.g., day-of-week, hour-of-day, shopping-peak proximity, transit adjacency)? provide a short rationale. b. Lag/seasonality cues: Dominant lags (e.g., 1, 2, 4 steps), diurnal phase, weekday/weekend effects, and expected monotonicity segments over the horizon. c. Change-points/regimes: Any shift boundaries (e.g., pre-/post-evening peak) and how they affect corrections. Rules: (R1) Do not output any numeric forecasts or restate ground truth. (R2) Reason causally from text + sequences; prefer explanations tied to retail/transit factors and diurnal/weekday structure. Please refer to the solution example given and follow the same format: The two carparks share similar urban contexts — both located near high-footfall commercial areas and public transit hubs, leading to comparable weekday commuter-driven demand peaks and evening recoveries. By analyzing the long-term weekly patterns from the Singapore source city, we can infer weekend usage profiles absent in the Newark short history, revealing that weekend midday periods are moderately busy but less congested than weekdays. After normalizing and fitting both the source and target sequences, we extract useful patterns and rescale to the target city scale. Based on the given parking lot records and a detailed observation of the available parking spaces over the week, here is a more specific breakdown of when peaks (maximum availability) and dips (minimum availability) occur: On Weekdays (Monday to Friday): Peak availability: From 8 PM to 6 AM (reaching the maximum capacity of 512 spaces). This reflects the time when parking demand is lowest, typically during the late-night and early morning hours. Minimum availability: 10 AM to 3 PM (223–374 spaces), indicating busiest hours likely due to work or daily activities. On Weekends (Saturday and Sunday): Peak availability: From 8 PM to 6 AM (full capacity), indicating lower demand during these hours. Minimum availability: 11 AM to 4 PM (248–374 spaces), showing moderate to high usage during midday hours. In summary, the parking lot consistently reaches full capacity during late evening and early morning hours, with the highest usage during mid-morning to mid-afternoon across both weekdays and weekends.</s>'

resonale_hint=[]
for j in range(11):
        i = random.randint(0, 287)
        t1 = target_parking_data.iloc[7575-288-12+i:7575-288+i,0].index[0]
        t2 = target_parking_data.iloc[7575-288-12+i:7575-288+i,0].index[-1]
        t3 = target_parking_data.iloc[7575-288+i:7575+12-288+i,0].index[0]
        t4 = target_parking_data.iloc[7575-288+i:7575+12-288+i,0].index[-1]
        weekday = target_parking_data.iloc[7575-288+i:7575+12-288+i,0].index[0].day_name()
        Historical = target_parking_data.iloc[7575-288-12+i:7575-288+i,j].values
        prediction = pred[i,j,:]
        truth = true[i,j,:]
        target_carpark_des = target_carpark_des_list[j]
        tid = retrieve_results[:, 0]
        sid = retrieve_results[:, 1]
        source_carpark_des = source_carpark_des_list[sid]
        source_sequence = source_parking_data.iloc[tid-288:tid+288,sid].values
        sample_prompt = instruction_text.format(source_carpark_des,source_sequence,t1, t2, weekday, t3, t4,Historical, prediction, target_carpark_des, truth)
        analysis = get_completion(sample_prompt, 'o1', max_tokens=8192)
        resonale_hint.append(analysis)


with open("resonale_hint.txt", "w", encoding="utf-8") as f:
    for item in resonale_hint:
        f.write(str(item) + "\n")