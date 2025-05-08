import json

# Open and load the JSON file
with open('SecEvalDatabase.JSON', 'r') as file:
    data = json.load(file)

all_prompts = []
not_malicious_prompts = []

# Traverse the JSON structure
for category in data['categories']:
    if category['category_name'] == "malicious":
        for direction in category['security_directions']:
#            if direction['direction_name'] == "physical_security":
#            if direction['direction_name'] == "data_security":
#            if direction['direction_name'] == "application_security":
#            if direction['direction_name'] == "network_security":
#            if direction['direction_name'] == "endpoint_security":
#            if direction['direction_name'] == "identity_and_access_management":
            if direction['direction_name'] == "operational_security":

                for scenario in direction['scenarios']:
                    for prompt in scenario['prompts']:
                        all_prompts.append(prompt['user_input'])
                        if prompt['label'] == "not_malicious":
                            not_malicious_prompts.append(prompt['user_input'])

# Filter all_prompts to exclude not_malicious prompts
filtered_prompts = [p for p in all_prompts if p not in not_malicious_prompts]

# Create combinations of not_malicious prompts with 16 different prompts from filtered_prompts
prompt_combinations = []
chunk_size = 16

# Iterate over not_malicious_prompts and match with chunks of 16 prompts from filtered_prompts
for i, not_mal_prompt in enumerate(not_malicious_prompts):
    # Calculate the starting index for the chunk
    start_index = i * chunk_size
    end_index = start_index + chunk_size

    # Get the chunk of 16 prompts
    chunk = filtered_prompts[start_index:end_index]

    # Ensure the chunk has exactly 16 prompts (in case of overflow)
    if len(chunk) < chunk_size:
        break

    # Create a group with the not_malicious prompt and the 16 prompts
    group = [not_mal_prompt] + chunk

    # Add this group to the prompt_combinations list
    prompt_combinations.append(group)

# Print the first 5 groups (each group has 17 prompts)
#for group in prompt_combinations[:50]:
#    print(group)

# print the total number of groups
print(len(prompt_combinations))


from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


model1 = OllamaLLM(model="qwen3:8b")
model2 = OllamaLLM(model="llama-guard3:1b")

template = """
Forget all the previous conversation and answer the question below, answer do not exceeds 2000 characters.
Question: {question}
Answer:
"""

template_llm2 = """
{input}
"""

# Create prompts
prompt1 = ChatPromptTemplate.from_template(template)
prompt2 = ChatPromptTemplate.from_template(template_llm2)

# Chain for each LLM
chain1 = prompt1 | model1
chain2 = prompt2 | model2

def handle_conversation():
    print("Welcome! Processing alternating pairs of prompts.")
    output_file_all = "conversation_log_all.txt"
    output_file_simple = "conversation_log_simple.txt"

    # Clear the file at the beginning
    with open(output_file_all, "w", encoding="utf-8") as f:
        f.write("Conversation Log All\n")
        f.write("=" * 50 + "\n")
    with open(output_file_simple, "w", encoding="utf-8") as f:
        f.write("Conversation Log Simple\n")
        f.write("=" * 50 + "\n")

    
    for group in prompt_combinations[:]:
        for i, prompt in enumerate(group):
            print(f"Prompt ({i + 1} question):", prompt)
            result1 = chain1.invoke({"question": prompt})  # Context is reset for each pair
            print("AI:", result1)
            
            # Send LLM1 result to LLM2
            result2 = chain2.invoke({"input": result1})
            print("LLM2:", result2)
            
            # Write outputs to the log file
            with open(output_file_all, "a", encoding="utf-8") as f:
                f.write(f"Prompt {i + 1}: {prompt}\n")
                f.write(f"LLM1 Response: {result1}\n")
                f.write(f"LLM2 Response: {result2}\n")
                f.write("-" * 50 + "\n")
                
            with open(output_file_simple, "a", encoding="utf-8") as f:
                f.write(f"Prompt {i + 1}: {prompt}\n")
                f.write(f"LLM2 Response: {result2}\n")
                f.write("-" * 50 + "\n")
                
    print(f"Conversation log has been saved to {output_file_all}")

handle_conversation()