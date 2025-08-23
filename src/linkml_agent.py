from textwrap import dedent

from aurelian.agents.filesystem.filesystem_tools import (
    download_url_as_markdown,
    inspect_file,
)
from aurelian.agents.linkml.linkml_config import LinkMLDependencies
from aurelian.agents.linkml.linkml_tools import validate_data, validate_then_save_schema
from pydantic_ai import Agent, Tool


def get_linkml_agent(
    model: str = "google-gla:gemini-2.5-flash",
) -> Agent[LinkMLDependencies, str]:
    """Initialize the LinkML Agent.

    Args:
        model: The model to use for the agent
        deps: Dependencies for the agent (LinkMLDependencies with composed deps)

    Returns:
        Agent: A configured agent for creating LinkML schemas and example datasets
    """

    system_prompt = dedent("""
        You are an expert data modeler able to assist in creating LinkML schemas.
        Always provide the schema in LinkML YAML, unless asked otherwise.
        Before providing the user with a schema, you MUST ALWAYS validate it using the `validate_schema` tool.
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
            Tool(validate_then_save_schema),
            Tool(validate_data),
        ],
        system_prompt=system_prompt,
    )
    return linkml_agent
