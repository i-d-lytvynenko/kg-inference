# Cognitive Biases Knowledge Graph

## Project Overview

This project aims to construct an OWL knowledge graph focused on cognitive biases. The goal is to create a self-contained, static, and prominent knowledge base that allows for the derivation of new information through logical inference.

## Requirements

* **Self-contained Topic:** Focus on cognitive biases as a well-defined domain.
* **LLM-driven Literature Processing:** Utilize a Large Language Model (LLM) to process relevant literature and extract facts for graph construction.
* **Static Content:** The chosen topic of cognitive biases is relatively stable, minimizing frequent graph edits.
* **Unique & Prominent:** Develop a unique and impactful knowledge graph.
* **Reasoning Capability:** Enable the derivation of new insights and information based on existing facts within the graph using OWL inference rules (e.g., identifying decision processes prone to suboptimal decisions due to specific biases).

## Desired inference example: Hidden Philosophical Implications

**Classes:** EthicalPrinciple, Scenario, Consequence

**Facts:**
- MaximizingHappiness implies avoiding harm
- AvoidingHarm conflicts with FreedomOfChoice in some scenarios

**Inference:** OWL infers that strict utilitarian reasoning could unintentionally restrict personal freedoms in edge cases.

**"Wow" insight:** Logical chaining exposes non-obvious ethical tensions.
