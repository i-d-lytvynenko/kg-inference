import asyncio
from pathlib import Path

from src.schema_agent import get_config, get_schema_agent
from src.utils import format_prompt


async def generate_schema():
    """
    Generates a LinkML schema for cognitive biases using the Schema agent.
    """
    agent = get_schema_agent()

    prompt = format_prompt("""
        Create a LinkML schema for cognitive biases. The schema should include:

        1.  **Entities:**
            *   `CognitiveBias`: A core entity representing a cognitive bias.
            *   `SystematicError`: A type of error that can be associated with cognitive biases.
            *   `DecisionProcess`: Represents a process where decisions are made.
            *   `SuboptimalDecision`: Represents a decision that is not optimal.

        2.  **Relationships:**
            *   `is_a`: Standard inheritance relationship (e.g., CognitiveBias is_a SystematicError).
            *   `leads_to`: Connects a cause to an effect (e.g., SystematicError leads_to SuboptimalDecision).
            *   `exhibits`: Connects a DecisionProcess to a CognitiveBias (e.g., DecisionProcess exhibits ConfirmationBias).
            *   `is_prone_to`: Connects a DecisionProcess to a SuboptimalDecision (this will be inferred).

        Ensure the schema is well-structured and includes appropriate types and slots for these entities and relationships.
    """)

    print("Generating LinkML schema...")
    deps = get_config()
    schema = (await agent.run(prompt, deps=deps)).output
    print("\n--- Generated LinkML Schema ---\n")
    print(schema)

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    schema_file = output_dir / "schema.yaml"
    schema_file.write_text(schema)
    print(f"\nSchema saved to {schema_file.absolute()}")


if __name__ == "__main__":
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("kg-inference")
    mlflow.pydantic_ai.autolog()  # pyright: ignore[reportPrivateImportUsage]

    with mlflow.start_run():
        asyncio.run(generate_schema())
