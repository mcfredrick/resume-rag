import dspy

OLLAMA_BASE = "http://matt-xps-8950:11434/v1"

student  = dspy.LM("openai/smollm2:latest", api_base=OLLAMA_BASE, api_key="ollama", max_tokens=300, temperature=0)
proposer = dspy.LM("openai/gemma3:12b",    api_base=OLLAMA_BASE, api_key="ollama", max_tokens=1024)
judge    = dspy.LM("openai/gemma3:12b",    api_base=OLLAMA_BASE, api_key="ollama", max_tokens=50)

dspy.configure(lm=student)


class PersonaRephrase(dspy.Signature):
    """Answer the question in character as the given persona, using only the provided facts."""
    persona:    str = dspy.InputField(desc="the character to speak as")
    query:      str = dspy.InputField(desc="the original question")
    raw_answer: str = dspy.InputField(desc="factual answer to draw from — do not invent new facts")
    response:   str = dspy.OutputField(desc="in-character response")


class Rephrase(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(PersonaRephrase)

    def forward(self, persona, query, raw_answer):
        return self.predict(persona=persona, query=query, raw_answer=raw_answer)


class JudgeScore(dspy.Signature):
    """Score a chatbot response 0-10.
    10 = strongly in-character voice AND all key facts from raw_answer preserved.
    0  = wrong persona voice OR key facts missing/wrong.
    Output only the integer."""
    persona:    str = dspy.InputField(desc="the persona the response should embody")
    raw_answer: str = dspy.InputField(desc="factual source material that must be preserved")
    response:   str = dspy.InputField(desc="the response to evaluate")
    score:      int = dspy.OutputField(desc="integer 0-10")


TRAINSET = [
    dspy.Example(
        persona="pirate",
        query="What programming languages does Matthew know?",
        raw_answer="Matthew's primary languages are Python, C++, and Rust. Python is his main language for AI/ML work; C++ for audio software and systems programming. He is also comfortable with JavaScript and web technologies.",
    ).with_inputs("persona", "query", "raw_answer"),
    dspy.Example(
        persona="sports announcer",
        query="What are Matthew's key achievements?",
        raw_answer="Matthew's key achievements: Shipped LLM safety guardrails protecting millions of Webex users. Architected the MLOps platform adopted as the org standard. Built a synthetic data pipeline reducing data-creation effort by 90%. Improved AI security classifiers by identifying mis-categorized edge cases. Mentored colleagues who advanced to new roles.",
    ).with_inputs("persona", "query", "raw_answer"),
    dspy.Example(
        persona="noir detective",
        query="What is Matthew's current role?",
        raw_answer="Matthew is a Senior Software Engineer on the AI Platform team at Cisco (Collaboration AI/Webex) since June 2022. He leads LLM safety, MLOps platform architecture, synthetic data pipelines, and AI agent development.",
    ).with_inputs("persona", "query", "raw_answer"),
    dspy.Example(
        persona="radio DJ",
        query="What are Matthew's hobbies and personal interests?",
        raw_answer="Outside of work, Matthew is passionate about woodworking, gardening, baking, and songwriting. He's also into home automation, DIY electronics, and is an award-winning hip hop dancer.",
    ).with_inputs("persona", "query", "raw_answer"),
    dspy.Example(
        persona="surfer dude",
        query="What ML frameworks and tools does Matthew use?",
        raw_answer="Matthew works with PyTorch, TensorFlow, scikit-learn, and Hugging Face Transformers. For MLOps: Weights & Biases, MLflow, Airflow. For infrastructure: AWS, Azure, Kubernetes, Docker, Pulumi, GitHub Actions.",
    ).with_inputs("persona", "query", "raw_answer"),
    dspy.Example(
        persona="pirate",
        query="What companies has Matthew worked for?",
        raw_answer="Cisco (Senior Software Engineer - AI Platform, June 2022–present); BandLab Technologies (Software Developer, November 2019–April 2022); DayJobDevelopment LLC (consultant, June 2019–present).",
    ).with_inputs("persona", "query", "raw_answer"),
    dspy.Example(
        persona="sports announcer",
        query="What ML and AI experience does Matthew have?",
        raw_answer="Matthew shipped LLM safety guardrails protecting millions of Webex users. Architected a comprehensive MLOps platform using Airflow and MLflow. Built synthetic data pipelines cutting data-creation effort by 90%. Developed autonomous red-teaming agents. Built computer vision pipelines for real-time video enhancement.",
    ).with_inputs("persona", "query", "raw_answer"),
    dspy.Example(
        persona="noir detective",
        query="What is Matthew's educational background?",
        raw_answer="Matthew earned a BS in Mechanical Engineering from Worcester Polytechnic Institute (WPI) in 2013. Certifications: Supervised Machine Learning, Advanced Learning Algorithms (Coursera), Advanced Audio Plugin Development (Kadenze).",
    ).with_inputs("persona", "query", "raw_answer"),
    # valset below
    dspy.Example(
        persona="radio DJ",
        query="What AI safety work has Matthew done?",
        raw_answer="Matthew designed and shipped LLM guardrails protecting millions of Webex users — content filtering, prompt injection defense, jailbreak prevention, and toxicity detection. He built autonomous red-teaming agents to stress-test AI systems.",
    ).with_inputs("persona", "query", "raw_answer"),
    dspy.Example(
        persona="surfer dude",
        query="Has Matthew done any mentorship or leadership work?",
        raw_answer="Matthew mentors colleagues through Cisco's formal mentorship program — mentees have advanced to new roles and owned major features end-to-end. He is the go-to technical authority for MLOps strategy and regularly briefs senior leadership.",
    ).with_inputs("persona", "query", "raw_answer"),
]

TRAINSET, VALSET = TRAINSET[:8], TRAINSET[8:]


def metric(example, pred, trace=None):
    with dspy.context(lm=judge):
        result = dspy.Predict(JudgeScore)(
            persona=example.persona,
            raw_answer=example.raw_answer,
            response=pred.response,
        )
    try:
        return int(str(result.score).strip().split()[0]) / 10.0
    except Exception:
        return 0.0


if __name__ == "__main__":
    # Sanity check before committing to a full optimization run
    print("=== Baseline sample ===")
    baseline = Rephrase()
    ex = TRAINSET[0]
    out = baseline(persona=ex.persona, query=ex.query, raw_answer=ex.raw_answer)
    print(f"Persona: {ex.persona}\nResponse: {out.response}\n")
    score = metric(ex, out)
    print(f"Judge score: {score:.1f}\n")

    print("=== Running MIPROv2 (instruction-only, 10 trials) ===")
    optimizer = dspy.MIPROv2(
        metric=metric,
        prompt_model=proposer,
        task_model=student,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        num_candidates=10,
        auto=None,
        verbose=True,
    )
    optimized = optimizer.compile(
        Rephrase(),
        trainset=TRAINSET,
        valset=VALSET,
        num_trials=15,
        minibatch_size=2,
        requires_permission_to_run=False,
    )

    print("\n" + "=" * 60)
    print("OPTIMIZED INSTRUCTION:")
    print("=" * 60)
    print(optimized.predict.signature.instructions)
    print("=" * 60)

    print("\n=== Optimized sample (same input) ===")
    out2 = optimized(persona=ex.persona, query=ex.query, raw_answer=ex.raw_answer)
    print(f"Response: {out2.response}")
    print(f"Judge score: {metric(ex, out2):.1f}")

    optimized.save("optimized_program.json")
    print("\nSaved to optimized_program.json")
