# pyright: reportUnknownVariableType = information, reportUnknownArgumentType = information
import io
from typing import cast

from linkml.generators.owlgen import MetadataProfile, OwlSchemaGenerator
from linkml_runtime.linkml_model import SchemaDefinition
from linkml_runtime.loaders import yaml_loader
from owlready2 import World, sync_reasoner_pellet

from src.config import get_settings

settings = get_settings()
linkml_schema_str = settings.schema_path.read_text()
rdf_triplets = settings.data_path.read_text()
linkml_schema = cast(
    SchemaDefinition,
    yaml_loader.loads(linkml_schema_str, target_class=SchemaDefinition),
)

gen = OwlSchemaGenerator(
    schema=linkml_schema,
    format="xml",
    use_native_uris=True,
    add_root_classes=False,
    add_ols_annotations=False,
    metaclasses=False,
    type_objects=False,
    assert_equivalent_classes=True,
    metadata_profiles=[MetadataProfile.rdfs],
)
owl_schema_str = gen.serialize()

world = World()
world.get_ontology(base_iri=linkml_schema.id).load(
    fileobj=io.BytesIO(str.encode(owl_schema_str))
)
world.get_ontology(base_iri="http://example.org/new_data#").load(
    fileobj=io.BytesIO(str.encode(rdf_triplets))
)

sync_reasoner_pellet(world, debug=2)
