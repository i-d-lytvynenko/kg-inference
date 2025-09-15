**"The Office Supplies & People Tracker"**

Imagine we're building a tiny system to keep track of a few things in a small office.
Our ontology is super basic, focusing on some common office items and the people who might use them.

Here's what our world looks like:

**1. Main "Categories" or "Types of Things" (Classes):**

*   **`Person`**: This is for any individual in our office. Think "human being."
*   **`OfficeItem`**: This is a very general category for anything that exists in the office that isn't a person.
*   **`WritingTool`**: This is a specific kind of `OfficeItem` that you use to write with.
*   **`ElectronicDevice`**: Another specific kind of `OfficeItem`, something that needs power or batteries.
*   **`Desk`**: A piece of furniture found in an office, where someone might work.

**2. How These Categories Relate (Class Hierarchies - "is a kind of"):**

*   A `WritingTool` **is a kind of** `OfficeItem`. (So, if something's a writing tool, it's definitely an office item.)
*   An `ElectronicDevice` **is a kind of** `OfficeItem`. (Similar to above, electronics are office items.)

**3. Ways Things Interact or Connect (Properties/Relationships):**

*   **`uses`**: This is a relationship between a `Person` and an `OfficeItem`.
    For example, "John `uses` a pen."
*   **`has`**: This is a relationship from a `Person` to a `Desk`.
    For example, "Alice `has` a desk."
*   **`locatedOn`**: This is a relationship from an `OfficeItem` to a `Desk`.
    For example, "The stapler `locatedOn` John's desk."
*   **`requiresPower`**: This is a characteristic or attribute that an `ElectronicDevice` *might* have.
    We'll treat this as a simple true/false property for now, but really it just flags that it's a thing that needs power.
    (This is almost a data property, but we can model it as an object property to a placeholder 'PowerSource' if we wanted,
     but for simplicity, let's think of it as a flag on `ElectronicDevice` for now.)

**4. Some Basic Rules or "Facts" About Our World (Axioms/Constraints - but keep it simple):**

*   **Implicit Rule 1 (from hierarchy):** If something is a `WritingTool`, it *must* also be an `OfficeItem`.
*   **Implicit Rule 2 (from hierarchy):** If something is an `ElectronicDevice`, it *must* also be an `OfficeItem`.

