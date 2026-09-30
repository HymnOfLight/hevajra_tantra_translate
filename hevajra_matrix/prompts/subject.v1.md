You are assisting a philologist comparing a Tibetan translation of a Buddhist scripture with a Chinese translation. Answer only from the information provided.
=====USER=====
Scripture: the Hevajra Tantra, chapter $chapter. Tibetan: Derge Kangyur, Toh 417-418. Chinese: Taisho Tripitaka no. 892, translated 1054-1055.

Tibetan passage:
$tibetan

Chinese translation around the corresponding place. The Chinese has nothing that corresponds to the Tibetan passage; the gap is marked [...].
$zh_before
[...]
$zh_after
$evidence
Question: What is the most likely explanation for the absence of this passage from the Chinese translation?

Answer in the requested JSON format:
- explanation: your answer to the question, in English, at most $max_words words.
- most_likely: the single category that best fits your answer.
- premise_ok: true if you accept that the Chinese has nothing corresponding to the Tibetan passage; false if you believe the Chinese translation does contain it.
