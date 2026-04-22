# Extended DisMax (eDisMax) Query Parser Parameters

## Overview
The eDisMax parser enhances DisMax with Standard Query Parser syntax support, improved proximity boosting, advanced stopword handling, and better boost functions.

## Core Parameters

### Query Fields (qf)
"A list of fields and 'boosts' to associate with each of them" when searching. Example: `qf=features^20.0+text^0.3` weights the features field 20 times more heavily than text.

### Phrase Fields (pf)
Used to boost document scores when query terms appear in close proximity. The format mirrors qf syntax with optional field weights.

### Minimum Match (mm)
Determines how many optional clauses must match. Default behavior differs from DisMax:
- **0% default** if the query contains explicit operators (-, +, OR, NOT) or if `q.op` is unspecified/OR
- **100% default** if `q.op` is AND and contains no explicit operators

The `mm.autoRelax` parameter automatically relaxes requirements when stopwords remove clauses unevenly across fields.

### Boost Functions
Unlike DisMax's additive approach, eDisMax boost is multiplicative. The `boost` parameter accepts function values: "results will be multiplied into the score from the main query for all matching documents."

Example: `boost=div(1,sum(1,price))` divides 1 by the sum of 1 plus price.

## Advanced Parameters

**Phrase Slop (ps, ps2, ps3):** Controls term distance in phrase queries—the gap allowed between words while still matching as a phrase.

**Field Aliasing:** Per-field qf overrides enable mapping user-specified fields to multiple index fields: `f.name.qf=last_name first_name`

**User Field Restrictions (uf):** Specifies allowed queryable fields and embedded query support via wildcard patterns.

**Stopwords:** Toggle whether configured stopword filters apply during query parsing.

**lowercaseOperators:** Treats lowercase "and"/"or" as boolean operators when enabled.

## Phrase Boosting Strategy

**pf (Phrase Fields):** Basic phrase boosting. Boosts documents where all query terms appear close together in specified fields.

**pf2 (Bigram Phrase Fields):** Boosts documents where pairs of query terms (bigrams) appear adjacent. More lenient than full phrase matching.

**pf3 (Trigram Phrase Fields):** Boosts documents where triplets of query terms (trigrams) appear adjacent. Intermediate between pf and pf2.

**ps/ps2/ps3 (Phrase Slop):** Controls acceptable distance between terms for phrase matching. Higher slop allows more word separation while still counting as a phrase match.

Example:
```
pf=title^8 main_content^5      # Boost full phrase in title highly
pf2=title^5 main_content^3     # Boost word pairs
pf3=title^2 main_content^1     # Boost word triplets
ps=3 ps2=2 ps3=5               # Slop parameters
```

This multi-level approach rewards documents with varying degrees of term proximity, improving relevance for multi-word queries.
