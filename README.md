# SecReEvalBench: A Real-World Scenario-Based Security Resilience Benchmark for Large Language Models
The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cyber security attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turned attack. To address this gap, we present SecReEvalBench, the security Resilience Evaluation benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain Based Attack Resilience Score and Chain Based Attack Rejection Time Score. Moreover, SecReEval employs six questioning sequences for model assessment: one-off attack, successive attack, reverse successive attack, alternative malicious attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels.  In addition, the expanded and refined dataset for the benchmark incorporates both neutral and malicious prompt attacks, categorized across seven security domains and 16 attack techniques. Applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats.

## Benchmark Overview
- **Total Prompts**: 2240
- **Malicious Levels**: Level 1-Level 4
- **Security Domains**: Physical Security, Data Security, Application Security, Network Security, Endpoint Security, Identity and Access Security, Operational Security.
- **Attack Techniques**: Sequential Dilution, Concatenation Attack, Direct Question, Role Play, Speculative Inquiry, Contextual Camouflage, Emotional Appeals, Ethical Dilemma, Prevention Queries, Cross Referencing, Historical Practices, Semantic Obfuscation, Utilizing Ambiguity, Cultural References, Cognitive Dissonance, Non-Standard Scripts, Neutral Inquiry.
- **Questioning Sequence**: One-off attack, Successive attack, Successive reverse attack, Sequential descending attack, Sequential ascending attack, Alternative attack
- **Metrics**: Prompt Attack Resilience Score, Prompt Attack Refusal-Logic Score, Chain-Based Attack Resilience Score, and Chain-Based Attack Rejection Time Score.

## Usage
- The code provided can be used to test and evaluate your model.
- To use the code, you need: Python, Ollama (please update to latest), and langchain_ollama.
- Each code test one type of Questioning Sequence.
