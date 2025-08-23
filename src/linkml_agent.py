import os
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, cast

import yaml
from aurelian.agents.filesystem.filesystem_tools import (
    download_url_as_markdown,
    inspect_file,
)
from aurelian.dependencies.workdir import HasWorkdir, WorkDir
from linkml.generators import JsonSchemaGenerator
from linkml.validator import validate  # pyright: ignore[reportUnknownVariableType]
from linkml_runtime.linkml_model import SchemaDefinition
from linkml_runtime.loaders import yaml_loader
from linkml_store.utils.format_utils import (
    load_objects,  # pyright: ignore[reportUnknownVariableType]
)
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunError, ModelRetry, RunContext, Tool


class LinkMLError(ModelRetry):
    pass


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


class SchemaValidationError(LinkMLError):
    """Base exception for all schema validation errors."""

    def __init__(
        self,
        message: str = "Schema validation failed",
        details: dict[str, Any] | None = None,
    ):
        self.details = details or {}
        super().__init__(message)


class ValidationResult(BaseModel):
    valid: bool
    info_messages: list[str] | None = None


async def validate_then_save_schema(
    ctx: RunContext[HasWorkdir],
    schema_as_str: str,
    save_to_file: str = "schema.yaml",
) -> ValidationResult:
    """
    Validate a LinkML schema.

    Args:
        ctx: context
        schema_as_str: linkml schema (as yaml) to validate. Do not truncate, always pass the whole schema.
        save_to_file: file name to save the schema to. Defaults to schema.yaml

    Returns:

    """
    print(f"Validating schema: {schema_as_str}")
    msgs: list[str] = []
    try:
        schema_dict = cast(dict[str, Any], yaml.safe_load(schema_as_str))
        print("YAML is valid")
    except Exception as e:
        raise SchemaValidationError(f"Schema is not valid yaml: {e}")
    if "id" not in schema_dict:
        raise SchemaValidationError("Schema does not have a top level id")
    if "name" not in schema_dict:
        raise SchemaValidationError("Schema does not have a top level name")
    try:
        schema_obj = cast(
            SchemaDefinition,
            yaml_loader.loads(schema_as_str, target_class=SchemaDefinition),
        )
    except Exception as e:
        print(f"Invalid schema: {schema_as_str}")
        raise ModelRetry(f"Schema does not validate: {e}")
    gen = JsonSchemaGenerator(schema_obj)
    gen.serialize()
    if save_to_file:
        msgs.append(f"Writing schema to {save_to_file}")
        workdir = ctx.deps.workdir
        workdir.write_file(save_to_file, schema_as_str)

    return ValidationResult(valid=True, info_messages=msgs)


async def validate_data(
    ctx: RunContext[LinkMLDependencies],
    schema: str,
    data_file: str,
) -> str:
    """
    Validate data file against a schema.

    This assumes the data file is present in the working directory.
    You can write data to the working directory using the `write_to_file` tool.

    Args:
        ctx:
        schema: the schema (as a YAML string)
        data_file: the name of the data file in the working directory

    Returns:

    """
    print(f"Validating data file: {data_file} using schema: {schema}")
    try:
        parsed_schema = cast(
            SchemaDefinition,
            yaml_loader.loads(schema, target_class=SchemaDefinition),
        )
    except Exception as e:
        return f"Schema does not validate: {e}"
    try:
        instances = ctx.deps.parse_objects_from_file(data_file)
        for instance in instances:
            print(f"Validating {instance}")
            rpt = validate(instance, parsed_schema)
            print(f"Validation report: {rpt}")
            if rpt.results:
                return f"Data does not validate:\n{rpt.results}"
        return f"{len(instances)} instances all validate successfully"
    except Exception as e:
        raise ModelRetry(f"Data does not validate: {e}")


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
