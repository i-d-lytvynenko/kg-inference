import asyncio

from src.config import get_settings
from src.knowledge_agent import get_config, get_knowledge_agent


async def extract_knowledge(document: str) -> None:
    settings = get_settings()
    agent = get_knowledge_agent(model=settings.knowledge_model_name)

    deps = get_config(data_path=settings.data_path, schema_path=settings.schema_path)
    prompt = f"Document to process:\n{document}"

    print("Extracting knowledge...")

    rdf_triplets = (await agent.run(prompt, deps=deps)).output
    print("Extracted triplets:")
    print(rdf_triplets)


if __name__ == "__main__":
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("kg-inference")
    mlflow.pydantic_ai.autolog()  # pyright: ignore[reportPrivateImportUsage]

    with open("input_examples/facts_1.md", "r") as f:
        document = f.read()

    with mlflow.start_run():
        asyncio.run(extract_knowledge(document=document))
