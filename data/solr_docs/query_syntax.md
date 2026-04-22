# Standard Query Parser: Key Concepts

## Field Boosting Syntax (^)

The caret symbol (`^`) increases a term's relevance weight. As the documentation states: "The higher the boost factor, the more relevant the term will be." For example, `jakarta^4 apache` prioritizes "jakarta" in search results. Boost factors must be positive but can be less than 1 (e.g., 0.2). Phrases can also be boosted: `"jakarta apache"^4`.

A related feature is constant scoring using `^=`, which sets a fixed score for matching documents regardless of term frequency or other relevancy factors.

## Query Operators

**Boolean operators** structure queries logically:

- **AND/&&**: Both terms required
- **OR/||**: Either term required (default operator)
- **NOT/!**: Excludes following term
- **+ (Required)**: Term must be present
- **- (Prohibit)**: Term must not be present

Per the guide: "When specifying Boolean operators with keywords such as AND or NOT, the keywords must appear in all uppercase."

## Phrase Queries

Phrases use double quotes and search for exact word sequences. For example, `"hello dolly"` finds documents with these words adjacent in that order.

## Relevance Scoring Principles

The standard parser bases relevance on term occurrence within documents. Factors include term frequency (how often it appears) and inverse document frequency (rarity across the index). Boosting allows manual adjustment of term importance, while constant-score queries override these automatic calculations for specific clauses.

## Field-Specific Queries

Queries can target specific fields using the syntax `field:value`. For example:
- `title:Solr` searches only the title field
- `title:"Apache Solr"` searches for the phrase in the title field
- `title:Solr^5 content:Solr` boosts title matches higher

This is particularly useful when you know certain fields are more important for specific query types.
