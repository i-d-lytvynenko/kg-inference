import os
from dataclasses import dataclass
from textwrap import dedent

from pydantic_ai import Agent, Tool

from src.tools import (
    HasWorkdir,
    WorkDir,
    download_url_as_markdown,
    inspect_file,
    validate_data,
    validate_then_save_schema,
)


@dataclass
class LinkMLDependencies(HasWorkdir):
    """Configuration for the LinkML agent."""

    pass


def get_config() -> LinkMLDependencies:
    """
    Get the LinkML agent configuration.

    Returns:
        LinkMLDependencies: The LinkML dependencies
    """

    workdir_path = os.environ.get("AURELIAN_WORKDIR", None)
    if workdir_path:
        return LinkMLDependencies(workdir=WorkDir(location=workdir_path))
    else:
        return LinkMLDependencies()


def get_linkml_agent(
    model: str = "google-gla:gemini-2.5-flash",
) -> Agent[LinkMLDependencies, str]:
    """Initialize the LinkML Agent.

    Args:
        model: The model to use for the agent

    Returns:
        Agent: A configured agent for creating LinkML schemas and example datasets
    """

    system_prompt = dedent("""
        You are an expert data modeler able to assist in creating LinkML schemas.
        Always provide the schema in LinkML YAML, unless asked otherwise.
        Before providing the user with a schema, you MUST ALWAYS validate it using the `validate_then_save_schema` tool.
        If there are mistakes, iterate on the schema until it validates.
        If it is too hard, ask the user for further guidance.
        If you are asked to make schemas for a file, you can look at files using
        the `inspect_file` tool.
        Always be transparent and show your working and reasoning. If you validate the schema,
        tell the user you did this.
        You should assume the user is technically competent, and can interpret both YAML
        schema files, and example data files in JSON or YAML.
    """)

    linkml_agent = Agent(
        model=model,
        deps_type=LinkMLDependencies,
        tools=[
            Tool(inspect_file),
            Tool(download_url_as_markdown),
            Tool(validate_then_save_schema, max_retries=3),
            Tool(validate_data, max_retries=3),
        ],
        system_prompt=system_prompt,
    )
    return linkml_agent
