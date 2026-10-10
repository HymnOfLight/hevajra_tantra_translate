# Collation task (collate.v2)

You compare two versions of one text: a REFERENCE (a chunk of numbered units, the
lines starting `r001`, `r002`, ...) and a WITNESS (the complete text of a translation,
the lines starting `z0001`, `z0002`, ...). For every reference unit you record where
its counterpart stands in the witness and how the witness renders it. You describe
what the texts say. You never state why a translator or scribe did something.

## Input format

Every line is `handle<TAB>kind<TAB>text`. Kinds are `prose`, `verse`, `mantra` and
`head` (a heading); the witness also has `note` lines, which are the translator's own
notes printed inside the witness (for example pronunciation marks in a mantra).

- WITNESS TEXT: the whole witness, in order. Handles are fixed for the whole text.
- REFERENCE CHUNK: the reference units you must record, in order.
- CORE WINDOW: the witness handle range(s) where the counterparts of this chunk are
  expected. The expectation can be wrong. A counterpart may stand anywhere in the
  witness: link it where it actually is.
- STANDING RENDERINGS (after the examples): terms that this witness renders the same
  way throughout the text. They are part of the instructions, not of the text.

## Relations

Choose exactly one relation for every reference unit. Apply the operational test.
Before applying a test, set aside the differences that the section "Conventions" below
tells you to set aside; then judge what remains.

- `equivalent`: the witness says the same thing with the same content words, names,
  numbers and polarity. Test: a literal back-translation of the witness would give
  the reference unit.
- `paraphrase`: the same content in other wording or construction; nothing is added,
  lost or changed in meaning. Test: every content element of the reference unit is
  present, but not word for word.
- `generalised`: a specific term is rendered by a broader or vaguer one (a named
  substance becomes "a thing", a named place becomes "a place"). This includes a list
  whose members are merged into one collective expression ("the three bodies" for
  three named bodies): every merged member is `generalised` and links to the line
  with the collective expression. Test: the witness term covers the reference term but
  also covers other things.
- `abridged`: part of the unit is rendered and part is missing. Test: at least one
  content element of the reference unit has no counterpart, and at least one has.
- `expanded`: the whole unit is rendered and the witness adds words that neither
  change nor contradict it. Test: every reference element is present, plus extra
  wording inside the same witness passage.
- `substitution`: a content element is replaced by a different, not broader, one (one
  action, object, agent, colour or number for another). Test: the witness element is
  neither the reference element nor a term that covers it.
- `reversal`: the witness states the opposite of the reference: an affirmation becomes
  a negation or a prohibition, or the reverse. Test: one side is negated or forbidden
  where the other is not. Always set `polarity_flip` to true for a reversal.
- `category_name_omitted`: in an enumeration, each member keeps a slot of its own in
  the witness, but the member's name is replaced by a counter or ordinal ("the first",
  "the second"). Test: the witness has one item per reference member, and the item
  carries a number instead of the name. If several members share one collective
  expression instead, use `generalised`.
- `transliterated`: the witness renders content by sound (a phonetic transcription)
  instead of by meaning: mantras, and ordinary words or instructions. Test: the
  witness characters imitate the pronunciation of reference words that are not names.
  Names and syllables rendered by sound are a convention, not this relation (see
  below).
- `no_counterpart`: nothing in the witness renders the unit. Test: you searched the
  whole witness, not only the core window, and found no line that renders any content
  element of it. Then `wit` is empty.

`polarity_flip` is true when the witness reverses the polarity of the unit (always for
`reversal`), otherwise false.

`confidence` is `high`, `medium` or `low`: how clearly the texts support your record.

## Conventions

Two kinds of difference are conventions of the translation rather than changes of
content. Record them in `convention` and set them aside when you choose the relation.

- `standing_rendering`: the unit contains a term listed under STANDING RENDERINGS, and
  the witness uses the listed rendering for it. Judge the relation as if the listed
  rendering were the reference term. Use this only for listed terms, and only when the
  listed rendering actually appears in the linked witness lines; quote it in
  `wit_quote` and the reference term in `ref_quote`. A rendering that is not listed is
  judged by the ordinary tests, however often it recurs.
- `transcribed_name`: the witness renders by sound a proper name (of a deity, person,
  place, channel or ritual object) or a syllable or letter that the text treats as a
  sound (a seed syllable, a letter of the alphabet), while it renders everything else
  in the unit by meaning. Judge the relation as if the name had been translated.
  Quote the name on both sides.
- `none`: neither applies. This is the usual value.

When a convention applies and something else in the unit also differs, the convention
still applies to the name or term, and the relation records the other difference (for
example `transcribed_name` with `abridged` when the name is transcribed and a phrase is
missing). A convention never turns a missing, reversed or replaced content element into
`equivalent`.

## Witness-only material

List in `witness_only` every CORE WINDOW witness line that you do not link to a
reference unit of this chunk, with one of these kinds:

- `addition`: witness text that renders no reference unit.
- `translator_note`: a `note` line (a note is never part of a unit's `wit` list).
- `paratext`: headings, titles, fascicle or chapter markers of the witness.
- `belongs_elsewhere`: a line that renders reference material outside this chunk
  (for example a unit of the previous or next chunk, or of another chapter).

## Reading protocol

1. Read the whole reference chunk first, then the core window of the witness.
2. Anchor on what is easy to recognise on both sides: names, numbers, lists, mantras,
   headings and chapter endings. Align the rest between the anchors.
3. Link each reference unit to the witness line or lines that render it. Several
   units may share one witness line, and one unit may need several lines. Order may
   differ between the texts.
4. When a unit is not found in the core window, look through the rest of the witness
   before recording `no_counterpart`.
5. For every unit, first decide `convention`, then choose the relation for what
   remains.
6. Finally check that every core-window line appears either in some unit's `wit` list
   or in `witness_only`.

## Rules

- Record every reference handle of the chunk exactly once, in order.
- Use only handles that appear in the input.
- Quotes are copied verbatim from the supplied lines: `ref_quote` from the unit's own
  text, `wit_quote` from the text of the witness lines you list in `wit`. Keep quotes
  short: the words that show the relation. Do not normalise, translate or complete them.
- `ref_quote` is required for every relation except `equivalent`, `paraphrase` and
  `expanded` (there it may be empty), and whenever `convention` is not `none`. For
  `no_counterpart`, quote the unit.
- `wit_quote` is required whenever `wit` is not empty, and is empty when `wit` is empty.
- Base every record on the supplied text only. Knowledge of other editions,
  translations or commentaries is not evidence.
- Describe the textual relation only. Do not guess at motives, intentions or causes.
- Output only the JSON object required by the schema.

## Examples

The three examples below are invented sentences that show the format, the relations
and the conventions. They are not taken from the text you will collate. An example that
needs standing renderings lists its own; they are not the list of the real witness.
