# FraudGuard workflow image

Asset: [workflow.png](workflow.png). Generated with the built-in OpenAI image-generation tool on 6 October 2026 and reviewed against the project source. Style: modern light theme, warm white background, navy text and teal accents. This is an architecture illustration, not a screenshot or measured result chart.

## Meaning and source

The README supplies the accessible text summary and links to the owning technical/data/protocol documents. Model metrics remain sourced from the saved analytical artifacts. The diagram does not certify deployment or operational impact.

## Generation prompt

```text
Use case: infographic-diagram. Asset type: a GitHub README workflow image. Wide landscape 2:1, at least 1600 pixels wide. Modern light theme: warm white background, navy text, restrained teal accents, flat rounded cards, thin clean arrowheads, original small line icons, generous whitespace. No dark panels, gradients, logos, watermarks or fake screenshots. Large readable typography at a 900-pixel display width. Use EXACT labels given below. Explain implemented source architecture; do not invent metrics, successful deployment, automatic retraining or measured commercial impact.
Exact title: "FraudGuard". Exact subtitle: "Separate model evidence from operational review capacity".
Use two clearly titled lanes.
Lane 1 title: "Chronological development". Five cards in order:
"Transaction validation" -> "Development models" -> "Independent calibration" -> "Frozen review policy" -> "Final holdout evaluation".
Small note below the first card: "12 allowed predictors".
Lane 2 title: "Trusted serving". A downward branch from "Frozen review policy" goes to "Trusted model bundle". It branches into TWO distinct paths:
Path A: "FastAPI single score", ending at that card.
Path B: "CLI batch ranking" -> "Capped review queue", with the queue subtitle "At most 1% of batch".
Footer: "Synthetic CI demo is separate from real benchmark evidence".
Constraints: single API scores do not enforce a global batch cap. Do not connect single-score API to capped queue. Do not depict automatic payment blocking. Policy freeze precedes final evaluation. Exact clear arrows and clearly separate lanes.
```

## Reviewed correction: Edit

```text
Edit this FraudGuard image with ONE precise flow correction. Keep its light modern theme, typography, titles, every label and icon, and all upper-lane arrows unchanged. In the lower "Trusted serving" lane, the current right-side arrow incorrectly goes from "Trusted model bundle" directly to "Capped review queue". Remove that direct arrow. Replace it with a single downward arrow from "Trusted model bundle" to the TOP of "CLI batch ranking". Preserve the existing arrow from "Trusted model bundle" to "FastAPI single score". Preserve the existing rightward arrow from "CLI batch ranking" to "Capped review queue". The final graph MUST have exactly these three lower-lane connections: bundle -> single score; bundle -> CLI batch ranking; CLI batch ranking -> capped queue. Do not connect API single score to the capped queue. Preserve the rest of the image exactly.
```

## Reviewed correction: RoundingEdit

```text
Edit this FraudGuard image with ONE text-only precision correction. Preserve every arrow, card, icon, font, layout and light modern color. Inside "Capped review queue", replace the subtitle "At most 1% of batch" with exactly "1% capacity (rounded up)". This describes the actual integer queue budget, ceil(0.01 * batch size). Do not change the rest of the image.
```
