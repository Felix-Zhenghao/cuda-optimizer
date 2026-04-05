## Motivation: Progressive Exposure

It is impossible and inefficient to dump a large documentation (e.g., NVIDIA's PTX ISA at 14 sections, 700+ subsections) into an agent's context window. The doc-crawl skill organizes crawled docs into a tree where an agent can **drill down** level by level:

1. Read a parent's README to see 20-word summaries of each child.
2. Pick interesting children and read their READMEs.
3. At a leaf, read the 50-word README summary. If it's worth it, read the full `doc.md`.

No one reads everything. Only interesting doc contents are loaded in the context window.

## The doc-crawl skill

This skill crawls a doc from the Internet and organizes it into a format for progressive exposure - each non-leaf node contains 20-word summary of all child nodes and each leaf node contains the doc content and a 50-word summary.

It does four steps:

1. Crawl the doc.
2. Merge short docs.
3. Split large docs.
4. Read docs and write summaries.

## The read-doc skill

This skill reads a doc following the progressive exposure principle.

It checks the doc format first and refuse to proceed if the doc is not organized correctly.

It forks exploration agents to select interesting leaf sections. Then it uses a verification agent to verify whether exploration results are truly interesting. If so, it asks the main agent to read the selected docs; else, it asks the exploration agent to explore further.

Everything is about saving the main agent's context window. The idea is similar to agent swarm as a context management strategy.