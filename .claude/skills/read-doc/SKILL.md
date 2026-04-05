---
name: read-doc
description: Read documentation from /doc using progressive exposure. Use when user asks about docs, wants to learn a topic, or asks "what docs do we have".
allowed-tools: Read, Glob, Grep
effort: low
model: sonnet
---

# Before you read: Check the doc format

Before reading, verify the target doc directory follows the expected format: leaf directories must contain a `doc.md` (full content) and a `README.md` (summary), and non-leaf directories must contain a `README.md` with child summaries. Glob for a few `doc.md` and `README.md` files to spot-check. If the format is wrong (e.g., missing READMEs, no `doc.md` in leaves, flat unstructured dumps), tell the user the doc is not in the expected format and stop. Do not attempt to read unstructured docs.

# Progressive Exposure Reading

The doc directory is organized as a tree. Each node has a `README.md` summary and leaf nodes have `doc.md` with full content. Follow the progressive exposure principle: start broad, drill down only where depth is needed.

## How to read:

- The progressive exposure principle means you should read the README summaries first to decide which sections to read in depth. Don't read everything. Only read interesting sections that may be relevant to user's query. For non-leaf nodes, read the README to get 20-word summaries of each child and select interesting children and read their READMEs. For leaf nodes, read the README for a 50-word summary before deciding to read the full `doc.md`.

- **Fork agents to explore what to read.** Fork exploration agents to explore according to the progressive exposure principle and return interesting leaf nodes to the verifier agent.

- **Verify interesting leaf nodes.** Fork a verifier agent to read all identified leaf nodes from exploration agents and verify if they are truly relevant to user's query. If all information is not enough to answer the query, tell the exploration agents to keep exploring other sections.

- **Read full content when necessary.** The main agent should only read the full `doc.md` of leaf nodes that are verified as relevant by the verifier agent.

- **Search when targeted.** If the user asks a specific question (e.g., "how does X work"), use Grep to find relevant `doc.md` files directly rather than walking the tree top-down.