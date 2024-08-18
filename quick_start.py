from openai import OpenAI
import os
from dotenv import load_dotenv

from prover.lean.verifier import Lean4ServerScheduler

# Load the environment variables from the .env file
load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

model_name = "llama3-70b-8192"

lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=1, timeout=300, memory_limit=10, name='verifier')

prompt = r'''Complete the following Lean 4 code. Do not include any code block markers. All explanations must be written in comments, only write the continuation to the code itself. Make sure to use Lean 4 syntax and not Lean 3: 
'''

code_prefix = r'''import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The second and fourth terms of a geometric sequence are $2$ and $6$. Which of the following is a possible first term?
Show that it is $\frac{2\sqrt{3}}{3}$.-/
theorem amc12b_2003_p6 (a r : ℝ) (u : ℕ → ℝ) (h₀ : ∀ k, u k = a * r ^ k) (h₁ : u 1 = 2)
  (h₂ : u 3 = 6) : u 0 = 2 / Real.sqrt 3 ∨ u 0 = -(2 / Real.sqrt 3) := by
'''

try:
    # Generate completion using the API
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt + code_prefix}],
        temperature=1.0,
        max_tokens=2048,
        top_p=0.95,
    )

    # Extract the generated code from the response
    generated_code = response.choices[0].message.content.strip()
    
    # Combine the code prefix with the generated code
    full_code = code_prefix + generated_code
    
    print("Generated Lean 4 code:")
    print(full_code)

    # Verify the generated proof
    request_id_list = lean4_scheduler.submit_all_request([full_code])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    print("\nVerification results:")
    print(outputs_list)

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    lean4_scheduler.close()