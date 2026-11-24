# Packaged similarity profile

`semantic.json` holds the default weights for the `semantic` tree when the MiniLM
layer is active: `description_minilm_centered_cosine=0.66`, `tfidf_cosine=0.34`,
fused with `variance_calibrated`. The pair was chosen in earlier AppWorld tool-tree
experiments. It contains no AppWorld data. Pass `semantic_weights=` or `profiles=`
to `OrchardBuilder` to use other weights.

OrchardLogic 0.1.0 ships no packaged taxonomy definitions, taxonomy heads, or
domain/function profiles.
