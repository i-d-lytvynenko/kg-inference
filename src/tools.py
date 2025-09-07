# pyright: reportUnknownVariableType = information
import contextlib
import io
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError

import requests
import yaml
from duckduckgo_search import DDGS
from jsonasobj2 import as_dict  # pyright: ignore[reportUnknownVariableType]
from linkml.generators.pythongen import PythonGenerator
from linkml.validator import validate
from linkml_runtime.linkml_model import SchemaDefinition
from linkml_runtime.loaders import rdflib_loader, yaml_loader
from linkml_runtime.utils.schemaview import SchemaView
from linkml_runtime.utils.yamlutils import YAMLRoot
from markdownify import markdownify
from oaklib.interfaces.search_interface import SearchInterface
from oaklib.selector import get_adapter
from owlready2 import (
    Ontology,
    OwlReadyInconsistentOntologyError,
    OwlReadyOntologyParsingError,
    get_ontology,
    sync_reasoner,
)
from pydantic import BaseModel
from pydantic_ai import ModelRetry, RunContext

logger = logging.getLogger(__name__)


@dataclass
class WorkDir:
    """
    Class to handle working directory operations.

    Example:

        >>> wd = WorkDir.create_temporary_workdir()
        >>> wd.check_file_exists("test.txt")
        False
        >>> wd.write_file("test.txt", "Hello, world!")
        >>> wd.check_file_exists("test.txt")
        True
        >>> wd.read_file("test.txt")
        'Hello, world!'
    """

    location: str = field(default_factory=tempfile.gettempdir)

    # NOTE:
    # The user is responsible for deleting the temporary directory
    # and its contents when done with it.
    @classmethod
    def create_temporary_workdir(cls) -> "WorkDir":
        temp_dir = tempfile.mkdtemp()
        return cls(location=temp_dir)

    def _ensure_location(self):
        location = Path(self.location)
        location.mkdir(parents=True, exist_ok=True)

    def __post_init__(self):
        self._ensure_location()

    def get_file_path(self, file_name: str) -> Path:
        self._ensure_location()
        return Path(self.location) / file_name

    def read_file(self, file_path: str) -> str:
        self._ensure_location()
        file_path = str(self.get_file_path(file_path))
        with open(self.get_file_path(file_path), "r") as f:
            return f.read()

    def check_file_exists(self, file_path: str) -> bool:
        self._ensure_location()
        return self.get_file_path(file_path).exists()

    def write_file(self, file_path: str, content: str) -> None:
        self._ensure_location()
        with open(self.get_file_path(file_path), "w") as f:
            f.write(content)

    def delete_file(self, file_path: str) -> None:
        self._ensure_location()
        self.get_file_path(file_path).unlink()

    def list_file_names(self) -> list[str]:
        """
        List the names of all files in the working directory.

        Note: the working directory is not recursively searched, it is flat

        Returns:
            list of file names
        """
        self._ensure_location()
        return [f.name for f in Path(self.location).iterdir() if f.is_file()]


@dataclass
class HasWorkdir:
    workdir: WorkDir = field(default_factory=lambda: WorkDir())


@dataclass
class HasSchema:
    schema_path: Path


@dataclass
class HasData:
    data_path: Path


async def search_web(query: str) -> str:
    """
    Search the web using a text query.

    Note, this will not retrieve the full content, for that you
    should use retrieve_web_page tool.

    Example:
        >>> result = web_search("Winner of 2024 nobel prize in chemistry")
        >>> assert "Baker" in result

    Args:
        query: Text query

    Returns:
        Matching web pages plus summaries
    """
    ddgs = DDGS()
    results = ddgs.text(query, max_results=10)
    if len(results) == 0:
        return "No results found! Try a less restrictive/shorter query."
    postprocessed_results = [
        f"[{result['title']}]({result['href']})\n{result['body']}" for result in results
    ]
    return "## Search Results\n\n" + "\n\n".join(postprocessed_results)


def retrieve_web_page(url: str) -> str:
    """Retrieve the text of a web page.

    Example:
        >>> url = "https://en.wikipedia.org/wiki/COVID-19"
        >>> text = retrieve_web_page(url)
        >>> assert "COVID-19" in text

    Args:
        url: URL of the web page

    Returns:
        str: The text of the web page

    """
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    markdown_content = markdownify(response.text).strip()

    # Remove multiple line breaks
    markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

    max_length = 10_000
    if len(markdown_content) <= max_length:
        return markdown_content
    else:
        return (
            markdown_content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + markdown_content[-max_length // 2 :]
        )


async def search_external_ontology(
    term: str,
    ontology: str,
    n: int = 10,
) -> list[tuple[str, str]]:
    """
    Search an external ontology for a term.

    Note that search should take into account synonyms, but synonyms may be incomplete,
    so if you cannot find a concept of interest, try searching using related or synonymous
    terms. For example, if you do not find a term for 'eye defect' in the Human Phenotype Ontology,
    try searching for "abnormality of eye" and also try searching for "eye" and then
    looking through the results to find the more specific term you are interested in.

    If you are searching for a composite term, try searching on the sub-terms to get a sense
    of the terminology used in the ontology.

    Args:
        term: The term to search for.
        ontology: The ontology ID to search. You can try prepending "ols:" to an ontology
        name to use the ontology lookup service (OLS), for example "ols:mondo" or
        "ols:hp". Try first using "ols:". You can also try prepending "sqlite:obo:" to
        an ontology name to use the local sqlite version of ontologies, but
        **you should prefer "ols:" because it seems to do better for finding
        non-exact matches!**
        n: The maximum number of results to return.

    Returns:
        A list of tuples, each containing an ontology ID and a label.
    """

    try:
        adapter = cast(SearchInterface, get_adapter(ontology))
        results = adapter.basic_search(term)
        results = list(adapter.labels(results))
    except (ValueError, URLError, KeyError):
        logger.warning(
            f"Unable to search ontology '{ontology}' - unknown url type: '{ontology}'"
        )
        return []
    if n:
        results = list(results)[:n]

    logger.info(f"Searched for '{term}' in '{ontology}' ontology")
    logger.info(f"Results: {results}")
    return results


async def search_project_ontology(
    ctx: RunContext[HasData],
    term: str,
    n: int = 10,
) -> list[tuple[str, str]]:
    """
    Search project ontology for a term.

    Note that search should take into account synonyms, but synonyms may be incomplete,
    so if you cannot find a concept of interest, try searching using related or synonymous
    terms. For example, if you do not find a term for 'eye defect' in the Human Phenotype Ontology,
    try searching for "abnormality of eye" and also try searching for "eye" and then
    looking through the results to find the more specific term you are interested in.

    If you are searching for a composite term, try searching on the sub-terms to get a sense
    of the terminology used in the ontology.

    Args:
        term: The term to search for.
        n: The maximum number of results to return.

    Returns:
        A list of tuples, each containing an ontology ID and a label.
    """
    return await search_external_ontology(
        term=term,
        ontology=str(ctx.deps.data_path),
        n=n,
    )


class ValidationResult(BaseModel):
    valid: bool
    info_messages: list[str]


async def inspect_file(ctx: RunContext[HasWorkdir], data_file: str) -> str:
    """
    Inspect a file in the working directory.

    Args:
        data_file: name of file

    Returns:
        ValidationResult
    """
    logger.info(f"Inspecting file: {data_file}")
    return ctx.deps.workdir.read_file(data_file)


async def validate_schema(
    schema_as_str: str,
) -> ValidationResult:
    """
    Validate a LinkML schema.

    Args:
        schema_as_str: linkml schema (as yaml) to validate. Do not truncate, always pass the whole schema.

    Returns:
        ValidationResult
    """
    logger.info(f"Validating schema: {schema_as_str}")
    msgs: list[str] = []
    try:
        schema_dict = cast(dict[str, Any], yaml.safe_load(schema_as_str))
        logger.info("YAML is valid")
    except Exception as e:
        msgs.append(f"Schema is not valid yaml: {e}")
        return ValidationResult(valid=False, info_messages=msgs)
    if "id" not in schema_dict:
        msgs.append("Schema does not have a top level id")
    if "name" not in schema_dict:
        msgs.append("Schema does not have a top level name")
    if msgs:
        return ValidationResult(valid=False, info_messages=msgs)
    try:
        _ = cast(
            SchemaDefinition,
            yaml_loader.loads(schema_as_str, target_class=SchemaDefinition),
        )
    except Exception as e:
        logger.error(f"Invalid schema: {schema_as_str}")
        msgs.append(f"Schema does not validate: {e}")
        return ValidationResult(valid=False, info_messages=msgs)
    return ValidationResult(valid=True, info_messages=msgs)


async def validate_data(
    ctx: RunContext[HasSchema],
    prefixes: str,
    data: list[tuple[str, str]],
) -> ValidationResult:
    """
    Validate data against the schema.

    Args:
        prefixes: Turtle prefixes used in any triplets in data.
        data: extracted data in Turtle format. Passed as a list of tuples ('ClassName', 'instance turtle code')

    Returns:
        ValidationResult
    """
    schema_path = ctx.deps.schema_path
    with open(schema_path, "r") as f:
        schema = f.read()

    logger.info(f"Validating data using schema: {schema}")
    parsed_schema = cast(
        SchemaDefinition,
        yaml_loader.loads(schema, target_class=SchemaDefinition),
    )

    gen = PythonGenerator(schema=parsed_schema)
    schemaview = SchemaView(schema=parsed_schema)
    python_module_str = gen.serialize()

    local_namespace = {}
    exec(python_module_str, local_namespace)  # pyright: ignore[reportUnknownArgumentType]

    for class_name, instance_source in data:
        try:
            target_class = local_namespace[class_name]  # pyright: ignore[reportUnknownVariableType]
        except KeyError:
            raise ModelRetry(f"Class not in schema: {class_name}")
        try:
            instance_source = prefixes + "\n\n" + instance_source
            instance = cast(
                YAMLRoot,
                rdflib_loader.loads(
                    source=instance_source,
                    target_class=target_class,  # pyright: ignore[reportUnknownArgumentType]
                    schemaview=schemaview,
                    allow_unprocessed_triples=False,
                ),
            )
            logger.info(f"Validating {instance}")
            rpt = validate(as_dict(instance), parsed_schema)
            logger.info(f"Validation report: {rpt}")
            if rpt.results:
                info_messages = [
                    f"{instance}: {m.message} ({m.type})" for m in rpt.results
                ]
                return ValidationResult(valid=False, info_messages=info_messages)
        except Exception as e:
            raise ModelRetry(f"Data does not validate: {e}")

    return ValidationResult(valid=True, info_messages=[])


# Because owlready2 is a joke
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        try:
            sys.stderr = devnull
            yield
        finally:
            sys.stderr = old_stderr


async def validate_owl_ontology(owl_content: str) -> ValidationResult:
    """
    Validate an OWL ontology for parsing errors and logical consistency.

    Args:
        owl_content: The OWL ontology content as a string in RDF/XML format.

    Returns:
        ValidationResult
    """
    logger.info("Validating OWL ontology.")
    msgs: list[str] = []

    owl_content = owl_content.strip()
    pattern = r"```(?:xml|owl|rdf)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, owl_content, re.IGNORECASE)
    if matches:
        owl_content = "\n\n".join(matches)

    fileobj = io.BytesIO(str.encode(owl_content))

    with suppress_stderr():
        try:
            onto = cast(
                Ontology,
                get_ontology(
                    base_iri="http://www.example.org/philosophical_implications#"
                ).load(fileobj=fileobj),
            )
            logger.info("Successfully loaded OWL ontology.")
        except OwlReadyOntologyParsingError as e:
            msg = f"Error parsing OWL content: {e}"
            logger.error(msg)
            msgs.append(msg)
            return ValidationResult(valid=False, info_messages=msgs)

        if logger.getEffectiveLevel() <= logging.DEBUG:
            classes = list(onto.classes())  # pyright: ignore[reportUnknownArgumentType]
            logger.debug(f"  - {len(classes)} OWL Classes found:")  # pyright: ignore[reportUnknownArgumentType]
            for cls in classes:
                logger.debug(
                    f"    - {cls.iri} (Label: {cls.label.first() if cls.label else 'N/A'})"
                )

            object_properties = list(onto.object_properties())  # pyright: ignore[reportUnknownArgumentType]
            logger.debug(f"  - {len(object_properties)} OWL Object Properties found:")  # pyright: ignore[reportUnknownArgumentType]
            for prop in object_properties:
                logger.debug(
                    f"    - {prop.iri} (Label: {prop.label.first() if prop.label else 'N/A'})"
                )

        try:
            with onto:
                sync_reasoner(debug=False)

            return ValidationResult(valid=True, info_messages=msgs)
        except OwlReadyInconsistentOntologyError:
            msg = "OWL ontology is logically inconsistent."
            logger.error(msg)
            msgs.append(msg)
            return ValidationResult(valid=False, info_messages=msgs)
