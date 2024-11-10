import os
import replicate

TOKEN_FILE = os.path.expanduser('~') + '/replicate_token_file.txt'

def read_token():
    with open(TOKEN_FILE, "r") as tf:
        return tf.readline()
    return ''

def run_with_prompt(input):
    output = replicate.run(
        "meta/meta-llama-3-8b-instruct",
        input=input
    )
    print("".join(output))

if __name__ == '__main__':
    os.environ["REPLICATE_API_TOKEN"] = read_token()
    #prompt = '''Classify: I saw a Gecko.
    #Sentiment: ?
    #Give one word response.
    #'''
    prompt = '''15 of us want to go to a restaurant.
Two of them have cars
Each car can seat 5 people.
Two of us have motorcycles.
Each motorcycle can fit 2 people.
Can we all get to the restaurant by car or motorcycle?
Think step by step.
Provide the answer as a single yes/no answer first.
Then explain each intermediate step.'''
    input = {
        "prompt": prompt,
        "max_new_tokens": 512,
        "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    }
    run_with_prompt(input)