import os
from dataclasses import dataclass
from textwrap import dedent
from typing import Any

from aurelian.agents.filesystem.filesystem_tools import (
    download_url_as_markdown,
    inspect_file,
)
from aurelian.agents.linkml.linkml_tools import validate_data, validate_then_save_schema
from aurelian.dependencies.workdir import HasWorkdir, WorkDir
from linkml_store.utils.format_utils import (
    load_objects,  # pyright: ignore[reportUnknownVariableType]
)
from pydantic_ai import Agent, AgentRunError, Tool


@dataclass
class LinkMLDependencies(HasWorkdir):
    """Configuration for the LinkML agent."""

    def parse_objects_from_file(self, data_file: str) -> list[dict[str, Any]]:
        """
        Parse objects from a file in the working directory.

        Args:
            data_file: Name of the data file in the working directory

        Returns:
            List of parsed objects
        """

        path_to_file = self.workdir.get_file_path(data_file)
        if not path_to_file.exists():
            raise AgentRunError(f"Data file {data_file} does not exist")
        return load_objects(path_to_file)


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
