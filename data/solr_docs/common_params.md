# Solr Common Query Parameters Summary

## Key Parameters for Search Results

**rows Parameter**
The `rows` parameter controls result set size, defaulting to 10 documents. It enables pagination by specifying how many matching documents to return per query.

**start Parameter**
This parameter sets an offset into results, beginning display from a specified position. Setting `start=3` skips the first three records, facilitating page-based navigation when combined with `rows`.

**fl (Field List) Parameter**
The `fl` parameter restricts returned information to specified fields. The default value `*` returns all stored fields or those with `docValues="true"` and `useDocValuesAsStored="true"`. You can specify space or comma-separated field names, use wildcards, add functions, or apply document transformers.

**defType Parameter**
This parameter selects the query parser processing the main `q` parameter. Options include `lucene` (default/Standard Query Parser), `dismax` (DisMax), `edismax` (Extended DisMax), or other available parsers.

## Additional Result-Shaping Parameters

**sort Parameter**
Controls result ordering by document score, field values, or function results. The default `score desc` sorts by relevance highest-to-lowest. Multiple sort criteria can be chained; subsequent criteria resolve ties.

**fq (Filter Query) Parameter**
Restricts the document superset without affecting scoring. Multiple `fq` parameters create intersecting document sets, and results cache independently, improving performance for repeated filters.

**wt Parameter**
Selects the response format writer, defaulting to JSON if unspecified.

## Best Practices

**Use fl to limit fields:** Only request fields you need. This reduces network overhead and speeds up queries.

**Cache filter queries:** Common filters should be in `fq` parameters, not `q`, to take advantage of Solr's filter cache. This dramatically improves performance for repeated filters.

**Choose appropriate defType:** Use `edismax` for user-facing searches where you need flexible query syntax and good relevance. Use `lucene` for precise, programmatic queries where you control the syntax.
