from pydantic_ai import Agent
from pydantic_ai.output import ToolOutput

from src.tools import validate_schema
from src.utils import format_prompt


class SchemaDependencies:
    """Configuration for the Schema agent."""

    pass


def get_config() -> SchemaDependencies:
    """
    Get the Schema agent configuration.

    Returns:
        SchemaDependencies: The Schema dependencies
    """

    return SchemaDependencies()


def get_schema_agent(
    model: str,
) -> Agent[SchemaDependencies, None]:
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
        Your final response should be ONLY the YAML content of the schema, without any additional text or markdown formatting.
    """)

    linkml_agent = Agent(
        model=model,
        deps_type=SchemaDependencies,
        output_type=ToolOutput(validate_schema, max_retries=5),
        system_prompt=system_prompt,
    )
    return linkml_agent
