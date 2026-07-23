# Limitations

- This is pre-alpha research software with one maintainer and no support SLA.
- Pilot models are 49.3M-parameter completion experiments with a 512-token
  context and no instruction tuning or safety alignment.
- Published pilot metrics lack complete raw, versioned evaluation provenance.
- Pilot model and tokenizer reuse terms remain unresolved.
- Web-derived data can contain private, copyrighted, biased, toxic, or otherwise
  harmful material despite filtering.
- Echelon has only architecture, tokenizer, and data-pipeline preflight evidence;
  no trained checkpoint or model evaluation exists.
- CPU latency, memory, energy, and quality have not been published under a
  reproducible protocol.
- Resume state can be unsafe when obtained from an untrusted source.
- Cross-platform and dependency-version differences can affect reproducibility.

Do not use released or locally trained artifacts for medicine, law, finance,
safety-critical control, consequential decisions, surveillance, or claims of
factual reliability.
