import logging
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai import RunContext

from src.tools import (
    HasData,
    HasSchema,
    HasWorkdir,
    WorkDir,
    inspect_file,
    logger,
    retrieve_web_page,
    search_external_ontology,
    search_project_ontology,
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
def mock_run_context_has_schema() -> RunContext[HasSchema]:
    mock_context = MagicMock(spec=RunContext)
    mock_context.deps = HasSchema(schema_path=Path("mock_schema.yaml"))
    return mock_context


@pytest.fixture
def mock_run_context_has_data() -> RunContext[HasData]:
    mock_context = MagicMock(spec=RunContext)
    mock_context.deps = HasData(data_path=Path("mock_data.ttl"))
    return mock_context


class TestWorkDir:
    def test_create_temporary_workdir(self) -> None:
        wd = WorkDir.create_temporary_workdir()
        assert Path(wd.location).is_dir()
        assert "tmp" in wd.location
        # Clean up the created directory
        shutil.rmtree(wd.location)

    def test_get_file_path(self, temp_workdir: WorkDir) -> None:
        file_path = temp_workdir.get_file_path("test.txt")
        assert file_path == Path(temp_workdir.location) / "test.txt"

    def test_write_and_read_file(self, temp_workdir: WorkDir) -> None:
        temp_workdir.write_file("test.txt", "Hello, world!")
        content = temp_workdir.read_file("test.txt")
        assert content == "Hello, world!"

    def test_check_file_exists(self, temp_workdir: WorkDir) -> None:
        assert not temp_workdir.check_file_exists("test.txt")
        temp_workdir.write_file("test.txt", "Hello, world!")
        assert temp_workdir.check_file_exists("test.txt")

    def test_delete_file(self, temp_workdir: WorkDir) -> None:
        temp_workdir.write_file("test.txt", "Hello, world!")
        assert temp_workdir.check_file_exists("test.txt")
        temp_workdir.delete_file("test.txt")
        assert not temp_workdir.check_file_exists("test.txt")

    def test_list_file_names(self, temp_workdir: WorkDir) -> None:
        temp_workdir.write_file("file1.txt", "content1")
        temp_workdir.write_file("file2.txt", "content2")
        files = temp_workdir.list_file_names()
        assert sorted(files) == ["file1.txt", "file2.txt"]


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
async def test_search_external_ontology() -> None:
    with patch("src.tools.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.basic_search.return_value = ["ID:1", "ID:2"]
        mock_adapter.labels.return_value = [("ID:1", "Label 1"), ("ID:2", "Label 2")]
        mock_get_adapter.return_value = mock_adapter
        results = await search_external_ontology("term", "ols:mondo")
        assert results == [("ID:1", "Label 1"), ("ID:2", "Label 2")]
        mock_get_adapter.assert_called_once_with("ols:mondo")
        mock_adapter.basic_search.assert_called_once_with("term")
        mock_adapter.labels.assert_called_once_with(["ID:1", "ID:2"])


@pytest.mark.asyncio
async def test_search_project_ontology(
    mock_run_context_has_data: RunContext[HasData],
) -> None:
    with patch(
        "src.tools.search_external_ontology", new_callable=AsyncMock
    ) as mock_search_external_ontology:
        mock_search_external_ontology.return_value = [("PROJ:1", "Project Term 1")]
        mock_run_context_has_data.deps.data_path = Path("my_ontology.owl")
        results = await search_project_ontology(
            mock_run_context_has_data, "project term"
        )
        assert results == [("PROJ:1", "Project Term 1")]
        mock_search_external_ontology.assert_called_once_with(
            term="project term", ontology="my_ontology.owl", n=10
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
async def test_validate_schema_valid() -> None:
    valid_schema = """
        id: http://example.com/test_schema
        name: test_schema
        classes:
          Person:
            attributes:
              name:
                range: string
    """
    result = await validate_schema(valid_schema)
    assert result.valid is True
    assert result.info_messages == []


@pytest.mark.asyncio
async def test_validate_schema_invalid_yaml() -> None:
    invalid_schema = """
        id: http://example.com/test_schema
          name: test_schema
    """
    result = await validate_schema(invalid_schema)
    assert result.valid is False
    assert result.info_messages is not None
    assert "Schema is not valid yaml" in result.info_messages[0]


@pytest.mark.asyncio
async def test_validate_schema_missing_id_name() -> None:
    invalid_schema = """
        classes:
          Person:
            attributes:
              name:
                range: string
    """
    result = await validate_schema(invalid_schema)
    assert result.valid is False
    assert result.info_messages is not None
    assert "Schema does not have a top level id" in result.info_messages
    assert "Schema does not have a top level name" in result.info_messages


@pytest.mark.asyncio
async def test_validate_data_valid(
    mock_run_context_has_schema: RunContext[HasSchema], temp_workdir: WorkDir
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
        @prefix : <https://w3id.org/linkml/examples/personinfo#> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
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
    schema_path = temp_workdir.get_file_path("mock_schema.yaml")
    temp_workdir.write_file(str(schema_path), schema_content)
    mock_run_context_has_schema.deps.schema_path = schema_path

    result = await validate_data(
        ctx=mock_run_context_has_schema,
        prefixes=prefixes,
        data=data,
    )
    assert result.valid


@pytest.mark.asyncio
async def test_validate_data_invalid(
    mock_run_context_has_schema: RunContext[HasSchema], temp_workdir: WorkDir
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
        @prefix : <https://w3id.org/linkml/examples/personinfo#> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
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
    schema_path = temp_workdir.get_file_path("mock_schema.yaml")
    temp_workdir.write_file(str(schema_path), schema_content)
    mock_run_context_has_schema.deps.schema_path = schema_path

    result = await validate_data(
        ctx=mock_run_context_has_schema,
        prefixes=prefixes,
        data=data,
    )
    assert not result.valid


@pytest.mark.asyncio
async def test_validate_owl_ontology_valid() -> None:
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
    result = await validate_owl_ontology(valid_owl)
    assert result.valid


@pytest.mark.asyncio
async def test_validate_owl_ontology_invalid_syntax() -> None:
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
    result = await validate_owl_ontology(invalid_owl)
    assert not result.valid


@pytest.mark.asyncio
async def test_validate_owl_ontology_inconsistent() -> None:
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
    result = await validate_owl_ontology(inconsistent_owl)
    assert not result.valid
