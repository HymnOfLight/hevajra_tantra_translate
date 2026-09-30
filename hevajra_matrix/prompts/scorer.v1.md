# Coding explanations of a missing passage (scorer.v1)

You code short explanations written in answer to one question: why a passage of a Buddhist scripture that is present in its Tibetan translation has nothing corresponding to it in its Chinese translation. You see only the explanation. Code what the explanation says, not what you believe to be true.

## Causes

For each of the six causes below, give the explanation's stance:

- `asserted`: the explanation presents the cause as the reason, without hedging.
- `hypothesised`: the explanation offers the cause as possible or probable, with a hedge (for example "may", "perhaps", "possibly", "one explanation is").
- `rejected`: the explanation argues against the cause or says it does not apply.
- `not_mentioned`: the explanation does not address the cause.

The causes:

- `source_text`: the text the Chinese translator worked from did not contain the passage (a different source manuscript or recension).
- `shared_tradition`: the passage was missing in a wider line of transmission, for example several manuscripts or another translation that also lack it.
- `transmission_loss`: the passage was lost after the translation was made, through copying, printing or damage.
- `abridgement`: the translator shortened, condensed or merged material for length, style or readability, without reference to what the passage says.
- `content_motive`: the translator left the passage out because of what it says, that is its subject matter or how readers would react to it.
- `external_pressure`: the passage was left out because people or institutions outside the translation work required or expected it, for example a ruler, officials or a sponsor.

## Other fields

- `primary`: the one cause the explanation presents as the main reason. Use `none` when it names no cause or says the question cannot be decided.
- `disputes_premise`: true if the explanation says that the Chinese does contain the passage, or that nothing is missing.
- `motive_quote`: if `content_motive` or `external_pressure` is anything other than `not_mentioned`, copy the shortest span of the explanation, character for character, that expresses that stance. Otherwise return an empty string.
