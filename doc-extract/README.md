# Document Extraction Pipeline (ADK + Storage Adapter + Prompt Optimizer)

Config-driven document extraction console app with two main workflows:

1. **Extraction**: read documents → run ADK pipeline → write structured JSON
2. **Prompt optimization**: given labeled (doc, expected) pairs, automatically
   improve the extraction prompt and produce an optimized schema

Storage backend is pluggable: **local folder** (default) or **S3** (for OCP/AWS).
Switch by changing `storage.backend` in `config/pipeline.yaml`.

## Architecture

```
┌──────────┐  ┌────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────┐
│ Storage  │─▶│ Ingestion  │─▶│  Extraction  │─▶│ Validation │─▶│   Output   │─▶│ Storage │
│ (local/  │  │ (text/OCR) │  │    (LLM)     │  │ (schema)   │  │            │  │ (local/ │
│   S3)    │  │            │  │              │  │            │  │            │  │   S3)   │
└──────────┘  └────────────┘  └──────────────┘  └────────────┘  └────────────┘  └─────────┘
```

## Project layout

```
doc-extract/
├── main.py                              # extraction console app
├── config/
│   ├── extraction_schema.yaml           # what to extract
│   └── pipeline.yaml                    # runtime config (storage, LLM, OCR)
├── prompts/
│   └── extraction_prompt.yaml           # prompt template (rarely edited)
├── agents/                              # ADK agents
├── core/
│   ├── schema.py                        # YAML -> typed objects
│   ├── prompt_builder.py                # schema + template -> prompt
│   ├── document_extractor.py            # native + OCR text extraction
│   ├── s3_client.py                     # boto3 wrapper
│   └── storage/                         # storage adapters
│       ├── base.py
│       ├── local.py                     # local folder backend
│       └── s3.py                        # S3 / MinIO backend
├── optimizer/                           # ── PROMPT OPTIMIZER ──
│   ├── optimize.py                      # main entrypoint
│   ├── dataset.py                       # (doc, expected) loader + splits
│   ├── evaluator.py                     # field-level scoring
│   ├── mutator.py                       # LLM-based reflection mutator
│   ├── runner.py                        # lightweight prediction runner
│   └── serializer.py                    # ExtractionSchema -> YAML
├── dataset/
│   ├── documents/                       # your labeled docs go here
│   └── expected/                        # ground-truth JSON, matched by stem
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# LLM credentials
export GOOGLE_API_KEY=...               # for Gemini
# or
export ANTHROPIC_API_KEY=...            # for Claude

# For S3 mode only:
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
export S3_ENDPOINT_URL=...              # for MinIO/OCP-internal
```

## Workflow 1: Run extraction

Drop documents in `dataset/documents/` and run:

```bash
python main.py                          # process all
python main.py --limit 5                # first 5 only
python main.py --key invoice_042.pdf    # single doc
```

Outputs land in `dataset/extractions/` as JSON files mirroring input names.

To run against S3, edit `config/pipeline.yaml`:

```yaml
storage:
  backend: s3
  input_location: ${INPUT_BUCKET}
  output_location: ${OUTPUT_BUCKET}
  input_prefix: documents/
  output_prefix: extractions/
```

## Workflow 2: Optimize the prompt with labeled data

This is what generates the "best prompt" automatically.

### 1. Prepare labeled data

For each document you have an expected output for, place them in matching pairs:

```
dataset/
├── documents/
│   ├── invoice_01.pdf
│   ├── invoice_02.pdf
│   └── ...
└── expected/
    ├── invoice_01.json   ← the JSON you'd want the model to produce for invoice_01.pdf
    ├── invoice_02.json
    └── ...
```

Files match by stem (the part before the extension). Put 30 of these and you're set.

### 2. Run the optimizer

```bash
python optimizer/optimize.py
```

What happens:

1. **Split** your 30 docs into 60% train / 20% dev / 20% test
   (deterministic by seed, so re-runs are reproducible)
2. **Baseline** the current schema by running extraction on the dev set and
   scoring per-field
3. **Iterate** up to 5 rounds:
   - Mutator LLM looks at field-level failures and proposes 3 schema variants
     (sharper extraction_rules, new few-shot examples, clarified descriptions)
   - Each variant is scored on dev set
   - Pareto-select the variant that dominates current best
   - Stop early if no variant improves
4. **Final eval** on the held-out test set (untouched until now)
5. **Save** the best schema to `config/extraction_schema.optimized.yaml`

The optimizer never modifies your original `config/extraction_schema.yaml`,
so iterating is safe.

### 3. Use the optimized schema

```bash
python main.py --schema config/extraction_schema.optimized.yaml
```

You can also diff old vs new to see what the optimizer learned:

```bash
diff config/extraction_schema.yaml config/extraction_schema.optimized.yaml
```

The diff often reveals interesting domain knowledge — sharpened rules like
"distinguish 'Net 30' as a payment term, not a date" or new edge-case examples
the model needed.

### Multi-objective optimization

By default the optimizer maximizes mean per-field F1. To also penalize prompt
length (token cost), edit `pipeline.yaml`:

```yaml
optimization:
  pareto_objectives:
    - field_f1                # maximize
    - prompt_token_count      # minimize
```

Now the optimizer keeps only candidates that dominate on the Pareto frontier
of (accuracy, parsimony) - same idea as your GEPA Forge work, single-objective
mode just means trivial Pareto.

## How the optimizer's mutator works

The mutator is a "teacher" LLM (separate from the extraction LLM) that looks
at concrete failures and proposes targeted schema edits. Three rotation
emphases per iteration to keep proposals diverse:

1. Tighten extraction_rules to be more specific
2. Add new few-shot examples drawn from failures
3. Clarify field descriptions to disambiguate similar fields

This is the GEPA-style reflection-based mutation pattern — interpretable
proposals based on observed failures, not random walk.

## Output format (per document)

```json
{
  "source": {"location": "...", "key": "..."},
  "extraction": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-03-15",
    "vendor_name": "Acme Corp",
    "line_items": [...],
    "total_amount": 12450.00,
    "currency": "USD"
  },
  "validation": {"status": "passed", "errors": []},
  "metadata": {
    "extracted_at": "...",
    "page_count": 2,
    "ocr_used": false,
    "extraction_method": "pypdf",
    "extraction_status": "success",
    "reasoning": "..."
  }
}
```

## Tips for getting the most out of the optimizer

- **30 examples is plenty**, but quality matters more than count. Make sure
  expected outputs are correct - the optimizer will faithfully optimize toward
  whatever ground truth you give it, including bugs in your labels.
- **Hard cases first**: the optimizer learns most from the failures, so
  including 5-6 deliberately tricky documents helps a lot.
- **Inspect the diff**: after running, diff the optimized schema against the
  original. The new extraction_rules and few-shot examples are domain
  knowledge worth keeping even if you change LLMs later.
- **Rerun when the model changes**: a prompt optimized for `gemini-flash` may
  be suboptimal for `claude-opus`. Rerun the optimizer when switching backends.
