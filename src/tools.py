import re
import tempfile
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError

import httpx
import requests
import yaml
from aurelian.utils.pubmed_utils import (
    doi_to_pmid,
    extract_doi_from_url,
    get_pmcid_text,
    get_pmid_text,
)
from duckduckgo_search import DDGS
from linkml.generators import JsonSchemaGenerator
from linkml.validator import validate  # pyright: ignore[reportUnknownVariableType]
from linkml_runtime.linkml_model import SchemaDefinition
from linkml_runtime.loaders import yaml_loader
from linkml_store.utils.format_utils import (
    load_objects,  # pyright: ignore[reportUnknownVariableType]
)
from markdownify import markdownify
from oaklib.interfaces.search_interface import SearchInterface
from oaklib.selector import get_adapter  # pyright: ignore[reportUnknownVariableType]
from pydantic import BaseModel
from pydantic_ai import AgentRunError, ModelRetry, RunContext


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

    location: str = field(default_factory=lambda: "/tmp")

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
class HasWorkdir(ABC):
    workdir: WorkDir = field(default_factory=lambda: WorkDir())


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

    PMCs are redirected:

        >>> url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5048378/"
        >>> text = retrieve_web_page(url)
        >>> assert "integrated stress response (ISR)" in text

    URLs with DOIs:

        >>> url = "https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-020-00889-8"
        >>> text = retrieve_web_page(url)
        >>> assert "photosynthesis" in text

    Args:
        url: URL of the web page

    Returns:
        str: The text of the web page

    """
    if url.startswith("https://pmc.ncbi.nlm.nih.gov/articles/PMC"):
        url = url.strip("/")
        pmc_id = url.split("/")[-1]
        # print(f"REWIRING URL: Fetching PMC ID: {pmc_id}")
        return get_pmcid_text(pmc_id)

    doi = extract_doi_from_url(url)
    if doi:
        pmid = doi_to_pmid(doi)
        if pmid is not None:
            return get_pmid_text(pmid)

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    # Convert the HTML content to Markdown
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


async def search_ontology_with_oak(
    term: str, ontology: str, n: int = 10, verbose: bool = True
) -> list[tuple[str, str]]:
    """
    Search an OBO ontology for a term.

    Note that search should take into account synonyms, but synonyms may be incomplete,
    so if you cannot find a concept of interest, try searching using related or synonymous
    terms. For example, if you do not find a term for 'eye defect' in the Human Phenotype Ontology,
    try searching for "abnormality of eye" and also try searching for "eye" and then
    looking through the results to find the more specific term you are interested in.

    Also remember to check for upper and lower case variations of the term.

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
        verbose: Whether to print debug information.

    Returns:
        A list of tuples, each containing an ontology ID and a label.
    """

    try:
        adapter = cast(SearchInterface, get_adapter(ontology))
        results = adapter.basic_search(term)
        results = list(adapter.labels(results))
    except (ValueError, URLError, KeyError):
        print(
            f"## TOOL WARNING: Unable to search ontology '{ontology}' - unknown url type: '{ontology}'"
        )
        return []
    if n:
        results = list(results)[:n]

    if verbose:
        print(f"## TOOL USE: Searched for '{term}' in '{ontology}' ontology")
        print(f"## RESULTS: {results}")
    return results


class LinkMLError(ModelRetry):
    pass


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


async def inspect_file(ctx: RunContext[HasWorkdir], data_file: str) -> str:
    """
    Inspect a file in the working directory.

    Args:
        ctx:
        data_file: name of file

    Returns:

    """
    print(f"Inspecting file: {data_file}")
    return ctx.deps.workdir.read_file(data_file)


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
    ctx: RunContext[HasWorkdir],
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
        path_to_file = ctx.deps.workdir.get_file_path(data_file)
        if not path_to_file.exists():
            raise AgentRunError(f"Data file {data_file} does not exist")
        instances = load_objects(path_to_file)

        for instance in instances:
            print(f"Validating {instance}")
            rpt = validate(instance, parsed_schema)
            print(f"Validation report: {rpt}")
            if rpt.results:
                return f"Data does not validate:\n{rpt.results}"
        return f"{len(instances)} instances all validate successfully"
    except Exception as e:
        raise ModelRetry(f"Data does not validate: {e}")


class DownloadResult(BaseModel):
    file_name: str
    num_lines: int


async def download_url_as_markdown(
    ctx: RunContext[HasWorkdir],
    url: str,
    local_file_name: str,
) -> DownloadResult:
    """
    Download contents of a web page.

    Args:
        ctx: context
        url: URL of the web page
        local_file_name: Name of the local file to save the

    Returns:
        DownloadResult: information about the downloaded file
    """
    workdir: WorkDir = ctx.deps.workdir
    from markdownify import markdownify

    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=20.0)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = markdownify(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        # Assuming write_file is also async
        workdir.write_file(local_file_name, markdown_content)

        return DownloadResult(
            file_name=local_file_name, num_lines=len(markdown_content.split("\n"))
        )
