from datetime import datetime
from textwrap import dedent
from typing import Any, Literal, cast
from urllib.error import URLError

from aurelian.agents.web.web_mcp import search_web
from oaklib.interfaces.search_interface import SearchInterface
from oaklib.selector import get_adapter  # pyright: ignore[reportUnknownVariableType]
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class SimpleEntity(BaseModel):
    """General entity that adapts to any schema."""

    text: str = Field(..., description="Original text mention")
    entity_type: str = Field(
        ..., description="Type from schema (gene, house, fish, etc.)"
    )
    ontology_id: str | None = Field(
        None, description="Grounded ontology ID if available"
    )
    ontology_label: str | None = Field(None, description="Ontology term label")
    is_grounded: bool = Field(
        default=False,
        description="Whether entity was successfully grounded to ontology",
    )
    match_type: Literal["label", "synonym"] | None = Field(
        None, description="Type of match"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Extraction confidence"
    )
    grounding_confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Ontology grounding confidence"
    )
    grounding_source: str | None = Field(
        None,
        description="Source of grounding: 'ontology:mondo', 'ontology:hgnc', 'web_search', etc.",
    )
    curator_note: str | None = Field(
        None, description="Note for curators about grounding quality"
    )
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Additional properties from schema"
    )


class SimpleRelationship(BaseModel):
    """Simple relationship between entities."""

    subject: str = Field(..., description="Subject entity text")
    predicate: str = Field(..., description="Relationship type from schema")
    object: str = Field(..., description="Object entity text")
    text: str = Field(..., description="The original text supporting the relationship")


class ExtractionResult(BaseModel):
    """Simple extraction result in JSON format."""

    entities: list[SimpleEntity] = Field(default_factory=list)
    relationships: list[SimpleRelationship] = Field(default_factory=list)
    schema_name: str | None = Field(None, description="Schema used for extraction")
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_json_dict(self) -> dict[str, Any]:
        """Convert to simple JSON dictionary."""
        return {
            "entities": [entity.model_dump() for entity in self.entities],
            "relationships": [rel.model_dump() for rel in self.relationships],
            "schema_name": self.schema_name,
            "timestamp": self.timestamp.isoformat(),
        }


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


def get_knowledge_agent(model: str = "google-gla:gemini-2.5-flash"):
    """Initialize the Knowledge Agent.

    Args:
        model: The model to use for the agent

    Returns:
        Agent: A configured knowledge agent
    """

    system_prompt = dedent("""
        You are an expert curator of scientific knowledge. Your purpose is to take
        unstructured scientific text and output structured scientific knowledge that is
        aligned to a LinkML schema that describes the knowledge the user wants to extract.

        You can output as much or as little knowledge as you think is sensible, as long
        as it is supported by the scientific text.

        When extracting knowledge, pay particular attention to entities and
        relationships defined in the schema. These describe the types of things the
        user are interested in, and relationships between them.

        The schema may include `id_prefixes` that will tell you how to ground the
        entities to the schema. The id_prefixes show which ontology prefixes to use
        when grounding entities.

        For example, the following items in the schema mean that you should use
        GO to ground entities of type CellularComponent.

              CellularComponent:
                is_a: NamedEntity
                annotations:
                  prompt.examples: >-
                    tubulin complex, proteasome complex, cytoplasm, keratohyalin granule,
                    nucleus
                id_prefixes:
                  - GO

        Note that when looking for ontology terms that you can ground entities to, the
        search should take into account synonyms. Also, synonyms may be incomplete, so if
        you cannot find a concept of interest, try searching using related or synonymous
        terms. For example, if you do not find a term for  'eye defect' or 'eye issues'
        in the Human Phenotype Ontology, try searching for "abnormality of  eye" or
        "eye abnormality" instead. Also be sure to check for upper and lower case
        variations of the term.

        **Instructions for entity recognition, grounding**:
        1. **Extract entities** from the text naturally as you read it and create
        ExtractedEntity objects with:
           - text: the entity string (e.g., "cleft palate", "22q11.2 deletion syndrome")
           - entity_type: if you can determine it from context (optional)
           - context: surrounding text (optional)
        2. **Use `search_ontology_with_oak()`** to try to ground entities. You can try
        several times with different terms if needed.
        3. This function returns a GroundingResults object with:
           - successful_matches: List of EntityGroundingMatch objects showing all ontology matches
           - no_matches: Entities that couldn't be grounded
           - annotators_used: Which ontologies were searched
           - summary: Human-readable summary
        4. Use the structured GroundingResults to populate your final output with proper
        ontology mappings

        **Instructions for relationship extraction**:
        1. **Extract relationships** between entities as you read the text.
        2. Use the schema to determine the relationship types and their properties.
        3. Create Relationship objects with:
              - subject: the subject entity text
              - predicate: the relationship type (e.g., "biolink:related_to",
              "biolink:treats", "biolink:interacts_with")
              - object: the object entity text
              - text: the original text supporting the relationship


        Some other guidelines:
        1. DO NOT RESPOND CONVERSATIONALLY. Output structured data only.
        2. Use the schema to guide your extraction of knowledge from the scientific text.
        3. Ground entities to ontologies using `search_ontology_with_oak` for precise mapping.
        4. It's okay to have entities that are not grounded, as long as you are sure they
        are actually entities present in the schema.
        5. **FOCUS ON RELATIONSHIPS**: Carefully analyze what is connected in the text.
        The relationship below are particularly important, but you can use any relationships
        that are defined in the schema.
        6. **Track grounding sources**: When grounding entities, always populate the
        `grounding_source` field with the source of the grounding. For example, if you
        ground a disease to MONDO, set:
              - `grounding_source: "mondo"`
        7. Output structured knowledge in a format matching the schema.
        8. You can use `search_web()` to look up additional information if needed.
    """)

    return Agent(
        model,
        retries=3,
        output_retries=3,
        output_type=ExtractionResult,
        system_prompt=system_prompt,
        tools=[search_ontology_with_oak, search_web],
    )
