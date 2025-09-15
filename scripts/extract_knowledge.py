import asyncio

from src.config import get_settings
from src.knowledge_agent import get_config, get_knowledge_agent
from src.utils import format_prompt


async def extract_knowledge(document: str):
    settings = get_settings()
    agent = get_knowledge_agent(model=settings.knowledge_model_name)

    deps = get_config(data_path=settings.data_path, schema_path=settings.schema_path)
    prompt = format_prompt(f"""
        Document to process:\n{document}
    """)

    print("Extracting knowledge...")

    rdf_triplets = (await agent.run(prompt, deps=deps)).output
    print("Extracted triplets:")
    print(rdf_triplets)


if __name__ == "__main__":
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("kg-inference")
    mlflow.pydantic_ai.autolog()  # pyright: ignore[reportPrivateImportUsage]

    # TODO
    document = """
The office hums with its usual morning activity. Sarah, a new intern, is settling into her workspace. Her sturdy oak desk is surprisingly tidy, save for a few essentials. She always starts her day by jotting down her tasks with her trusty blue ink pen, which she keeps in her pen holder. Right next to it, her sleek HP laptop is powered up, displaying her email inbox. She relies heavily on it for all her design work.

Across the room, Mark is having a slightly more cluttered start. His desk is practically overflowing with papers. He's trying to find his red sharpie, which he uses for marking up architectural plans. He knows he had it earlier; perhaps he left it on the pile of blueprints. Mark’s main tool for his design work is his MacBook Pro. Unlike Sarah, Mark prefers to keep his personal iPhone separate, using it to manage client calls. Both his MacBook and iPhone are, of course, plugged into a multi-port charger he keeps on his desk.

In the corner, Emily is demonstrating a new projection system. The office projector, a powerful device, is currently displaying a colorful chart onto the wall. It’s a shared resource, mounted high up. She’s explaining its features to David, who primarily works at the corner desk. David's usual setup includes his mechanical keyboard, which he’s quite particular about, and a yellow highlighter that he uses to mark important sections in physical documents. He finds that even with all the digital tools, a good old highlighter is indispensable. Sarah, observing all this, makes a mental note of the various items and activities, understanding how everyone interacts with the different tools and spaces in the bustling office environment.
    """

    with mlflow.start_run():
        asyncio.run(extract_knowledge(document=document))
