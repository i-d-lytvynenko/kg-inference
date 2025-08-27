import os
from dataclasses import dataclass

from pydantic_ai import Agent, Tool

from src.tools import (
    HasWorkdir,
    WorkDir,
    download_url_as_markdown,
    inspect_file,
    validate_data,
    validate_schema,
)
from src.utils import format_prompt


@dataclass
class SchemaDependencies(HasWorkdir):
    """Configuration for the Schema agent."""

    pass


def get_config() -> SchemaDependencies:
    """
    Get the Schema agent configuration.

    Returns:
        SchemaDependencies: The Schema dependencies
    """

    workdir_path = os.environ.get("WORKDIR", None)
    if workdir_path:
        return SchemaDependencies(workdir=WorkDir(location=workdir_path))
    else:
        return SchemaDependencies()


def get_schema_agent(
    model: str,
) -> Agent[SchemaDependencies, str]:
    """Initialize the Schema Agent.

    Args:
        model: The model to use for the agent

    Returns:
        Agent: A configured agent for creating LinkML schemas
    """

    system_prompt = format_prompt("""
        You are an expert data modeler able to assist in creating LinkML schemas.
        Always provide the schema in LinkML YAML.
        Before providing the user with a schema, you MUST ALWAYS validate it using the `validate_schema` tool.
        If there are mistakes, iterate on the schema until it validates.
        If you are asked to make schemas for a file, you can look at files using the `inspect_file` tool.
    """)

    linkml_agent = Agent(
        model=model,
        deps_type=SchemaDependencies,
        tools=[
            Tool(inspect_file),
            Tool(download_url_as_markdown),
            Tool(validate_schema, max_retries=5),
            Tool(validate_data, max_retries=3),
        ],
        system_prompt=system_prompt,
    )
    return linkml_agent
