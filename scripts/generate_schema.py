import asyncio

from src.config import get_settings
from src.schema_agent import get_config, get_schema_agent


async def generate_schema(document: str) -> None:
    """
    Generates a LinkML schema using the Schema agent.
    """
    settings = get_settings()
    agent = get_schema_agent(model=settings.schema_model_name)

    prompt = f"Document to process:\n{document}"

    print("Generating LinkML schema...")

    deps = get_config(schema_path=settings.schema_path)
    schema = (await agent.run(prompt, deps=deps)).output
    print("Generated LinkML Schema:")
    print(schema)

if __name__ == "__main__":
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("kg-inference")
    mlflow.pydantic_ai.autolog()  # pyright: ignore[reportPrivateImportUsage]

    with open("input_examples/schema_description.md", "r") as f:
        document = f.read()

    with mlflow.start_run():
        asyncio.run(generate_schema(document=document))
