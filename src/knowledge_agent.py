from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.output import ToolOutput

from src.dependencies import HasData
from src.tools import (
    lookup_external_ontology_terms,
    lookup_project_ontology_terms,
    retrieve_web_page,
    search_web,
    validate_data,
    validate_owl_ontology,
)
from src.utils import format_prompt


class KnowledgeDependencies(HasData):
    """Configuration for the Knowledge agent."""

    pass


def get_config(data_path: Path, schema_path: Path) -> KnowledgeDependencies:
    """
    Get the Knowledge agent configuration.

    Returns:
        KnowledgeDependencies: The Knowledge dependencies
    """

    return KnowledgeDependencies(data_path=data_path, schema_path=schema_path)


def get_knowledge_agent(model: str) -> Agent[KnowledgeDependencies, str]:
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
        output_type=ToolOutput(validate_owl_ontology, max_retries=5),
        system_prompt=system_prompt,
        tools=[
            lookup_external_ontology_terms,
            lookup_project_ontology_terms,
            search_web,
            retrieve_web_page,
            validate_data,
        ],
    )
