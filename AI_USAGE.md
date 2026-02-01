# GenAI Usage Logs - CLS Group 7


### Copilot (Claude Sonnet 3.5)
* Aid with generating docstrings

### Claude App
* Debug package dependency issues
* Create doc strings for functions

<br>

---

## Matteo Postiferi

### 1. Summary
I utilized Generative AI as a support tool functioning for coding tasks and for conceptual refinement. While  AI assisted with syntax and structure, I retained full responsibility for the model's logic, validity, and scientific accuracy.

### 2. Tooling Strategy 
I used different models based on their specific strengths:


I leveraged **ChatGPT / Gemini** for initial brainstorming to map the paper's theoretical rules into programmable logic (identifying parameters, defining scenarios) and **Claude (3.5 Sonnet)** to draft initial code snippets and in the end transform my initial prototyping notebook into a clean  Python module (`soc_model.py`). Claude was particularly useful for its strict adherence to coding standards (PEP-8).


**Prompting Strategy Used:**
By assigning the AI a specific role (e.g., "Senior Python Engineer" or "Complex Systems Researcher"), I constrained its output to be strictly technical and explanatory, avoiding "black-box" solutions.

### 3. Workflow

**Phase A: Implementation**

* **Goal:** Implement the time-delayed buffer update rule described in the paper.
* **GenAI Output:** The AI generated draft snippets that updated node buffers synchronously and instantaneously.
* **My Evaluation:** I critically assessed the code and identified that it violated the "local rules" of the model. The AI failed to account for the specific time delay required for the "memory" effect.
* **Decision:** I rejected the AI's logic but kept the syntactic structure. I implemented the corrected version manually.
* **Reasoning:** Without this modification, the simulation produced flat time series, failing to reproduce the emergent properties of SOC.

**Phase B: Refactoring**
* **Goal:** Clean up the prototyping notebook into a reusable module.
* **GenAI Output:** A `soc_model.py` file with type hints and docstrings.
* **My Evaluation:** The code was logically equivalent to my validated prototype but significantly more readable and strictly typed.
* **Decision:** I accepted the structural changes as they improved code quality without altering the scientific behavior.

#### 4. Specific Prompts Provided (Examples)

**Prompt for Refactoring (Claude):**
> "Act as a Senior Python Engineer. Take this monolithic simulation function [PASTE CODE] and refactor it into a stateless module adhering to PEP-8. Do not alter the variable names (alpha, beta, epsilon) as they correspond to the paper's mathematical notation. Ensure inputs are clearly typed."


#### 5. Effectiveness & Limitations
* **Effectiveness:** The AI was highly effective at **syntax generation** and **library management**, significantly reducing the time spent on boilerplate code and formatting (PEP-8).
* **Limitations:** The AI demonstrated poor "contextual awareness" regarding specific scientific constraints. It repeatedly defaulted to global/synchronous updates (easier to code) rather than the required local/asynchronous rules. 