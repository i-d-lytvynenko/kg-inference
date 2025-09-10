# pyright: reportUnknownVariableType = information
import contextlib
import io
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, cast
from urllib.error import URLError

import requests
import yaml
from duckduckgo_search import DDGS
from jsonasobj2 import as_dict  # pyright: ignore[reportUnknownVariableType]
from linkml.generators.owlgen import OwlSchemaGenerator
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
    World,
    sync_reasoner_pellet,
)
from pydantic_ai import ModelRetry, RunContext

from src.dependencies import HasData, HasSchema, HasWorkdir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    title: str
    href: str
    summary: str


async def search_web(query: str) -> list[SearchResult]:
    """
    Search the web using a text query.

    If you get no results, try a less restrictive/shorter query.
    Note, this will not retrieve the full page content.

    Example:
        >>> result = web_search("Winner of 2024 nobel prize in chemistry")
        >>> assert "Baker" in result

    Args:
        query: Text query

    Returns:
        Matching web pages plus summaries
    """
    ddgs = DDGS()
    search_results: list[SearchResult] = []
    for result in ddgs.text(query, max_results=10):
        search_results.append(
            SearchResult(
                title=result.get("title", "N/A"),
                href=result.get("href", "N/A"),
                summary=result.get("body", "N/A"),
            )
        )

    return search_results


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


async def lookup_external_ontology_terms(
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
        raise ModelRetry(
            f"Unable to search ontology '{ontology}' - unknown url type: '{ontology}'"
        )
    if n:
        results = list(results)[:n]

    logger.info(f"Searched for '{term}' in '{ontology}' ontology")
    logger.info(f"Results: {results}")
    return results


async def lookup_project_ontology_terms(
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
    return await lookup_external_ontology_terms(
        term=term,
        ontology=str(ctx.deps.data_path),
        n=n,
    )


async def inspect_file(ctx: RunContext[HasWorkdir], data_file: str) -> str:
    """
    Inspect a file in the working directory.

    Args:
        data_file: name of file

    Returns:
        File content
    """
    logger.info(f"Inspecting file: {data_file}")
    return ctx.deps.workdir.read_file(data_file)


async def validate_schema(
    schema_as_str: str,
) -> None:
    """
    Validate a LinkML schema and save if successful.

    Args:
        schema_as_str: linkml schema (as yaml) to validate. Do not truncate, always pass the whole schema.

    Returns:
        None
    """
    logger.info(f"Validating schema: {schema_as_str}")
    try:
        schema_dict = cast(dict[str, Any], yaml.safe_load(schema_as_str))
        logger.info("YAML is valid")
    except Exception as e:
        raise ModelRetry(f"Schema is not valid yaml: {e}")
    msgs: list[str] = []
    if "id" not in schema_dict:
        msgs.append("Schema does not have a top level id")
    if "name" not in schema_dict:
        msgs.append("Schema does not have a top level name")
    if msgs:
        raise ModelRetry("\n".join(msgs))
    try:
        _ = cast(
            SchemaDefinition,
            yaml_loader.loads(schema_as_str, target_class=SchemaDefinition),
        )
    except Exception as e:
        logger.info(f"Invalid schema: {schema_as_str}")
        msgs.append(f"Schema does not validate: {e}")
        raise ModelRetry(f"Schema does not validate: {e}")


async def validate_data(
    ctx: RunContext[HasSchema],
    prefixes: str,
    data: list[tuple[str, str]],
) -> None:
    """
    Validate data against the schema and save if successful.

    Args:
        prefixes: Turtle prefixes used in any triplets in data.
        data: extracted data in Turtle format. Passed as a list of tuples ('ClassName', 'instance turtle code')

    Returns:
        None
    """
    linkml_schema_str = ctx.deps.schema_path.read_text()

    logger.info(f"Validating data using schema: {linkml_schema_str}")
    linkml_schema = cast(
        SchemaDefinition,
        yaml_loader.loads(linkml_schema_str, target_class=SchemaDefinition),
    )

    gen = PythonGenerator(schema=linkml_schema)
    schemaview = SchemaView(schema=linkml_schema)
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
            rpt = validate(as_dict(instance), linkml_schema)
            logger.info(f"Validation report: {rpt}")
            if rpt.results:
                info_messages = [
                    f"{instance}: {m.message} ({m.type})" for m in rpt.results
                ]
                raise ModelRetry("\n".join(info_messages))
        except Exception as e:
            raise ModelRetry(f"Data does not validate: {e}")


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


async def validate_owl_ontology(
    ctx: RunContext[HasData],
    rdf_triplets: str,
) -> None:
    """
    Validate new data against an OWL ontology for logical consistency and save if successful.

    Args:
        rdf_triplets: Data content as a string in RDF/XML format.

    Returns:
        None
    """
    logger.info("Validating OWL ontology.")

    rdf_triplets = rdf_triplets.strip()
    pattern = r"```(?:xml|owl|rdf)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, rdf_triplets, re.IGNORECASE)
    if matches:
        rdf_triplets = "\n\n".join(matches)

    linkml_schema_str = ctx.deps.schema_path.read_text()

    prev_rdf_triplets: str | None = None
    if ctx.deps.data_path.exists():
        prev_rdf_triplets = ctx.deps.data_path.read_text()

    linkml_schema = cast(
        SchemaDefinition,
        yaml_loader.loads(linkml_schema_str, target_class=SchemaDefinition),
    )

    # schemaview = SchemaView(schema=linkml_schema)
    gen = OwlSchemaGenerator(schema=linkml_schema)
    owl_schema_str = gen.serialize()

    with suppress_stderr():
        try:
            world = World()
            onto = cast(
                Ontology,
                world.get_ontology(base_iri=linkml_schema.id).load(
                    fileobj=io.BytesIO(str.encode(owl_schema_str))
                ),
            )

            if prev_rdf_triplets is not None:
                onto.load(fileobj=io.BytesIO(str.encode(prev_rdf_triplets)))

            onto.load(fileobj=io.BytesIO(str.encode(rdf_triplets)))
            logger.info("Successfully loaded OWL ontology with new data.")
        except OwlReadyOntologyParsingError as e:
            raise ModelRetry(f"Error parsing OWL content: {e}")

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
                sync_reasoner_pellet(onto, debug=2)

        except OwlReadyInconsistentOntologyError as e:
            error_message = str(e)
            marker = "\nThis is the output of `pellet explain`:"
            start_index = error_message.find(marker)
            if start_index != -1:
                error_message = error_message[start_index + len(marker) :]

            raise ModelRetry(
                f"OWL ontology is logically inconsistent. Pellet output: {error_message}."
            )

    if prev_rdf_triplets is not None:
        ctx.deps.data_path.write_text(prev_rdf_triplets + "\n\n" + rdf_triplets)
    else:
        ctx.deps.data_path.write_text(rdf_triplets)
