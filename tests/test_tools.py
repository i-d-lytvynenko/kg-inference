import logging
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import ModelRetry, RunContext

from src.dependencies import (
    HasData,
    HasSchema,
    HasWorkdir,
    WorkDir,
)
from src.tools import (
    inspect_file,
    logger,
    lookup_external_ontology_terms,
    lookup_project_ontology_terms,
    retrieve_web_page,
    search_web,
    validate_data,
    validate_owl_ontology,
    validate_schema,
)

logger.setLevel(logging.DEBUG)


@pytest.fixture
def temp_workdir() -> Generator[WorkDir, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield WorkDir(location=tmpdir)


@pytest.fixture
def mock_run_context_has_workdir(temp_workdir: WorkDir) -> RunContext[HasWorkdir]:
    mock_context = MagicMock(spec=RunContext)
    mock_context.deps = HasWorkdir(workdir=temp_workdir)
    return mock_context


@pytest.fixture
def mock_run_context_has_schema(temp_workdir: WorkDir) -> RunContext[HasSchema]:
    mock_context = MagicMock(spec=RunContext)
    schema_path = Path(temp_workdir.location) / "mock_schema.yaml"
    mock_context.deps = HasSchema(schema_path=schema_path)
    return mock_context


@pytest.fixture
def mock_run_context_has_data(
    mock_run_context_has_schema: RunContext[HasSchema], temp_workdir: WorkDir
) -> RunContext[HasData]:
    mock_context = MagicMock(spec=RunContext)
    data_path = Path(temp_workdir.location) / "mock_data.ttl"
    mock_context.deps = HasData(
        data_path=data_path, schema_path=mock_run_context_has_schema.deps.schema_path
    )
    return mock_context


@pytest.mark.asyncio
async def test_search_web() -> None:
    with patch("src.tools.DDGS") as mock_ddgs:
        mock_instance = mock_ddgs.return_value
        mock_instance.text.return_value = [
            {
                "title": "Test Title 1",
                "href": "http://example.com/1",
                "body": "Test Body 1",
            },
            {
                "title": "Test Title 2",
                "href": "http://example.com/2",
                "body": "Test Body 2",
            },
        ]
        search_results = await search_web("test query")
        assert len(search_results) == 2


def test_retrieve_web_page() -> None:
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><body><h1>Test</h1><p>This is a test page.</p></body></html>"
        )
        mock_get.return_value = mock_response
        result = retrieve_web_page("http://example.com")
        assert "Test" in result
        assert "This is a test page." in result
        assert "<html>" not in result

    # Test truncation
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        long_text = "a" * 15000
        mock_response.text = f"<html><body><p>{long_text}</p></body></html>"
        mock_get.return_value = mock_response
        result = retrieve_web_page("http://example.com/long")
        assert "..._This content has been truncated" in result
        assert len(result) < 11000  # Max length 10000 + some for truncation message


@pytest.mark.asyncio
async def test_lookup_external_ontology_terms() -> None:
    with patch("src.tools.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.basic_search.return_value = ["ID:1", "ID:2"]
        mock_adapter.labels.return_value = [("ID:1", "Label 1"), ("ID:2", "Label 2")]
        mock_get_adapter.return_value = mock_adapter
        results = await lookup_external_ontology_terms("term", "ols:mondo")
        assert results == [("ID:1", "Label 1"), ("ID:2", "Label 2")]
        mock_get_adapter.assert_called_once_with("ols:mondo")
        mock_adapter.basic_search.assert_called_once_with("term")
        mock_adapter.labels.assert_called_once_with(["ID:1", "ID:2"])


@pytest.mark.asyncio
async def test_lookup_project_ontology_terms(
    mock_run_context_has_data: RunContext[HasData],
) -> None:
    with patch(
        "src.tools.lookup_external_ontology_terms", new_callable=AsyncMock
    ) as mock_lookup_external:
        mock_lookup_external.return_value = [("PROJ:1", "Project Term 1")]
        mock_run_context_has_data.deps.data_path = Path("my_ontology.owl")
        results = await lookup_project_ontology_terms(
            mock_run_context_has_data, "project term"
        )
        assert results == [("PROJ:1", "Project Term 1")]
        mock_lookup_external.assert_called_once_with(
            term="project term", ontology=str(Path("my_ontology.owl")), n=10
        )


@pytest.mark.asyncio
async def test_inspect_file(
    mock_run_context_has_workdir: RunContext[HasWorkdir],
) -> None:
    mock_run_context_has_workdir.deps.workdir.write_file(
        "test_file.txt", "File content"
    )
    content = await inspect_file(mock_run_context_has_workdir, "test_file.txt")
    assert content == "File content"


@pytest.mark.asyncio
async def test_validate_schema_valid(
    mock_run_context_has_schema: RunContext[HasSchema],
) -> None:
    valid_schema = """
        id: http://example.com/test_schema
        name: test_schema
        imports:
          - linkml:types
        classes:
          Person:
            attributes:
              name:
                range: string
    """
    await validate_schema(ctx=mock_run_context_has_schema, schema_as_str=valid_schema)


@pytest.mark.asyncio
async def test_validate_schema_invalid_yaml(
    mock_run_context_has_schema: RunContext[HasSchema],
) -> None:
    invalid_schema = """
        id: http://example.com/test_schema
          name: test_schema
    """
    with pytest.raises(ModelRetry, match="Schema is not valid yaml"):
        await validate_schema(ctx=mock_run_context_has_schema, schema_as_str=invalid_schema)

@pytest.mark.asyncio
async def test_validate_schema_missing_id_name(
    mock_run_context_has_schema: RunContext[HasSchema],
) -> None:
    invalid_schema = """
        classes:
          Person:
            attributes:
              name:
                range: string
    """
    with pytest.raises(ModelRetry) as e:
        await validate_schema(ctx=mock_run_context_has_schema, schema_as_str=invalid_schema)
    assert "Schema does not have a top level id" in str(e.value)
    assert "Schema does not have a top level name" in str(e.value)


@pytest.mark.asyncio
async def test_validate_data_valid(
    mock_run_context_has_schema: RunContext[HasSchema],
) -> None:
    schema_content = """
        id: https://w3id.org/linkml/examples/personinfo
        name: personinfo
        prefixes:
          linkml: https://w3id.org/linkml/
          personinfo: https://w3id.org/linkml/examples/personinfo/
          ORCID: https://orcid.org/
        imports:
          - linkml:types
        default_prefix: personinfo
        default_range: string

        classes:
          Person:
            attributes:
              id:
                identifier: true
              full_name:
                required: true
                description:
                  name of the person
              age:
                range: integer
                minimum_value: 0
                maximum_value: 120
    """
    prefixes = """
        @prefix ORCID: <https://orcid.org/> .
        @prefix personinfo: <https://w3id.org/linkml/examples/personinfo/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    """

    data = [
        (
            "Person",
            """
                ORCID:1234 a personinfo:Person ;
                    personinfo:age 30 ;
                    personinfo:full_name "Clark Kent" .
            """,
        )
    ]
    mock_run_context_has_schema.deps.schema_path.write_text(schema_content)

    await validate_data(
        ctx=mock_run_context_has_schema,
        prefixes=prefixes,
        data=data,
    )


@pytest.mark.asyncio
async def test_validate_data_invalid(
    mock_run_context_has_schema: RunContext[HasSchema],
) -> None:
    schema_content = """
        id: https://w3id.org/linkml/examples/personinfo
        name: personinfo
        prefixes:
          linkml: https://w3id.org/linkml/
          personinfo: https://w3id.org/linkml/examples/personinfo/
          ORCID: https://orcid.org/
        imports:
          - linkml:types
        default_prefix: personinfo
        default_range: string

        classes:
          Person:
            attributes:
              id:
                identifier: true
              full_name:
                required: true
                description:
                  name of the person
              age:
                range: integer
                minimum_value: 0
                maximum_value: 120
    """
    prefixes = """
        @prefix ORCID: <https://orcid.org/> .
        @prefix personinfo: <https://w3id.org/linkml/examples/personinfo/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    """

    data = [
        (
            "Person",
            """
                ORCID:1234 a personinfo:Person ;
                    personinfo:age 330 ;
                    personinfo:full_name "Clark Kent" .
            """,
        )
    ]
    mock_run_context_has_schema.deps.schema_path.write_text(schema_content)

    with pytest.raises(ModelRetry, match="Data does not validate"):
        await validate_data(
            ctx=mock_run_context_has_schema,
            prefixes=prefixes,
            data=data,
        )


@pytest.mark.asyncio
async def test_validate_owl_ontology_valid(
    mock_run_context_has_data: RunContext[HasData],
) -> None:
    schema_content = """
        id: http://www.example.org/philosophical_implications#
        name: philosophical_implications
        prefixes:
          linkml: https://w3id.org/linkml/
          philosophical_implications: http://www.example.org/philosophical_implications#
        imports:
          - linkml:types
        default_prefix: philosophical_implications
        default_range: string

        classes:
          Concept:
            attributes:
              id:
                identifier: true
    """
    mock_run_context_has_data.deps.schema_path.write_text(schema_content)

    valid_owl = """
        <rdf:RDF xmlns="http://www.example.org/philosophical_implications#"
            xml:base="http://www.example.org/philosophical_implications#"
            xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:owl="http://www.w3.org/2002/07/owl#"
            xmlns:xml="http://www.w3.org/XML/1998/namespace"
            xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
            xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
            <owl:Ontology rdf:about="http://www.example.org/philosophical_implications#"/>
            <owl:Class rdf:about="http://www.example.org/philosophical_implications#Concept"/>
        </rdf:RDF>
    """
    await validate_owl_ontology(mock_run_context_has_data, valid_owl)


@pytest.mark.asyncio
async def test_validate_owl_ontology_invalid_syntax(
    mock_run_context_has_data: RunContext[HasData],
) -> None:
    schema_content = """
        id: http://www.example.org/philosophical_implications#
        name: philosophical_implications
        default_prefix: philosophical_implications
        classes:
          Concept:
            attributes:
              id:
                identifier: true
    """
    mock_run_context_has_data.deps.schema_path.write_text(schema_content)
    invalid_owl = """
        <rdf:RDF xmlns="http://www.example.org/philosophical_implications#"
            xml:base="http://www.example.org/philosophical_implications#"
            xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:owl="http://www.w3.org/2002/07/owl#"
            xmlns:xml="http://www.w3.org/XML/1998/namespace"
            xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
            xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
            <owl:Ontology rdf:about="http://www.example.org/philosophical_implications#"/>
            <owl:Class rdf:about="http://www.example.org/philosophical_implications#Concept"/>
        </rdf:RDF
    """  # Missing closing tag
    with pytest.raises(ModelRetry, match="Error parsing OWL content"):
        await validate_owl_ontology(mock_run_context_has_data, invalid_owl)


@pytest.mark.asyncio
async def test_validate_owl_ontology_inconsistent(
    mock_run_context_has_data: RunContext[HasData],
) -> None:
    schema_content = """
        id: http://www.example.org/philosophical_implications#
        name: philosophical_implications
        default_prefix: philosophical_implications
        classes:
          Concept:
            attributes:
              id:
                identifier: true
    """
    mock_run_context_has_data.deps.schema_path.write_text(schema_content)
    inconsistent_owl = """
        <rdf:RDF xmlns="http://www.example.org/philosophical_implications#"
            xml:base="http://www.example.org/philosophical_implications#"
            xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:owl="http://www.w3.org/2002/07/owl#"
            xmlns:xml="http://www.w3.org/XML/1998/namespace"
            xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
            xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
            <owl:Ontology rdf:about="http://www.example.org/philosophical_implications#"/>
            <owl:Class rdf:about="http://www.example.org/philosophical_implications#Concept">
                <owl:equivalentClass>
                    <owl:Class>
                        <owl:complementOf rdf:resource="http://www.example.org/philosophical_implications#Concept"/>
                    </owl:Class>
                </owl:equivalentClass>
            </owl:Class>
        </rdf:RDF>
    """
    with pytest.raises(ModelRetry, match="OWL ontology is logically inconsistent"):
        await validate_owl_ontology(mock_run_context_has_data, inconsistent_owl)
