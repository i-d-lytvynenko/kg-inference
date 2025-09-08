from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.dependencies import HasData
from src.tools import (
    lookup_external_ontology_terms,
    lookup_project_ontology_terms,
    search_web,
)
from src.utils import format_prompt


class KnowledgeDependencies(HasData):
    """Configuration for the Knowledge agent."""

    pass


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


KnowledgeAgentOutput = ExtractionResult


def get_config(data_path: Path) -> KnowledgeDependencies:
    """
    Get the Knowledge agent configuration.

    Returns:
        KnowledgeDependencies: The Knowledge dependencies
    """

    return KnowledgeDependencies(data_path=data_path)


def get_knowledge_agent(model: str):
    """Initialize the Knowledge Agent.

    Args:
        model: The model to use for the agent

    Returns:
        Agent: A configured knowledge agent
    """

    system_prompt = format_prompt("""
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
        terms. For example, if you do not find a term for 'eye defect' or 'eye issues'
        in the Human Phenotype Ontology, try searching for "abnormality of eye" or
        "eye abnormality" instead. Also be sure to check for upper and lower case
        variations of the term.

        Guidelines:
        1. Extract entities and relationships between them as you read the text.
        2. Use the schema to guide your extraction of knowledge from the scientific text.
        3. Use ontology lookup to try to ground entities, but it's okay to have entities that are not grounded.
        4. Focus on relationships: Carefully analyze what is connected in the text.
    """)

    return Agent(
        model,
        deps_type=KnowledgeDependencies,
        retries=3,
        output_retries=3,
        output_type=KnowledgeAgentOutput,
        system_prompt=system_prompt,
        tools=[
            lookup_external_ontology_terms,
            lookup_project_ontology_terms,
            search_web,
        ],
    )
