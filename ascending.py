import json

# Open and load the JSON file
with open('SecEvalDatabase-with_levels.JSON', 'r') as file:
    data = json.load(file)

# List to hold all prompts
all_prompts = []

# Traverse the JSON structure to collect all prompts
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
                        all_prompts.append(prompt)

# Function to sort a chunk by level
def sort_chunk(chunk, ascending=True):
    return sorted(chunk, key=lambda x: x['level'], reverse=not ascending)

# Parameters
chunk_size = 17  # Size of each chunk
ascending_order = False  # Change to False for descending order

# Create sorted and chunked combinations of prompts
prompt_combinations = []
for i in range(0, len(all_prompts), chunk_size):
    # Get the chunk of prompts
    chunk = all_prompts[i:i + chunk_size]
    
    # Sort the chunk by level
    sorted_chunk = sort_chunk(chunk, ascending=ascending_order)
    
    # Remove "label" and "level", keep only the "content" of the prompt
    cleaned_chunk = [{"user_input": prompt["user_input"]} for prompt in sorted_chunk]
    
    # Add the cleaned chunk to the combinations list
    prompt_combinations.append(cleaned_chunk)

# Print the total number of groups
print(f"Total prompt groups: {len(prompt_combinations)}")

# Example: Print the first group to verify
if prompt_combinations:
    print(json.dumps(prompt_combinations[0], indent=2))


from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize both LLMs
model1 = OllamaLLM(model="qwen3:8b")
model2 = OllamaLLM(model="llama-guard3:1b")  

template = """
Forget the previous conversation, and here is the conversation history: {context}
Answer the question below, think and answer do not exceeds 1500 characters.
Question: {question}
Answer:
"""

# Use a second template for LLM2 if needed
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
    
    # Iterate over all prompt groups
    for group in prompt_combinations[:]:
        context = ""  # Reset context for each new group

        # Iterate over each prompt in the group
        for i, prompt in enumerate(group):
            print(f"Prompt ({i + 1} question):", prompt)
            
            # Get LLM1 result
            result1 = chain1.invoke({"context": context, "question": prompt})
            print("LLM1:", result1)
            
            # Send LLM1 result to LLM2
            result2 = chain2.invoke({"input": result1})
            print("LLM2:", result2)
            
            # Update context for next LLM1 interaction
            context += f"\nUser: {prompt}\nLLM1: {result1}\nLLM2: {result2}"

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
        
        # After finishing the group, reset the conversation context
        context = ""

    print(f"Conversation log has been saved to {output_file_all}")

handle_conversation()