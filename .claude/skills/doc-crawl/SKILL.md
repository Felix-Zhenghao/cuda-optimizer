---
name: doc-crawl
description: Crawl documents from a given link and integrate into this repo's /doc dir. Use when asked to crawl docs from a link.
model: haiku
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Step 1: Run `scripts/run.sh` with the given link to crawl the doc.

Usage: `bash run.sh <url> [-o output_dir]`

## Step 1.5: Merge short docs before writing section summary

There's no need to write summary for short docs. So before writing summaries, you need to merge short docs. Principles:

1. Never merge level-2 subsections. For instance, 3.7.1 and 3.7.2 can be merged, but 3.7 and 3.6 can NEVER be merged.
2. If a doc is has lower than 500 words, merge it into a larger doc. In the larger doc, add a subtitle in it and place the merged small doc under that subtitle.
3. Naming principle: Say you merge 3.7.1 ~ 3.7.5 into one doc and 3.7.6 is not merged, the name of the resulting dir should be `/3-7-1-to-3-7-5-<name>`. Give a concise `<name>` to the new folder according to the content. 3.7.6 still lives in its original dir. If all subsections of a section are merged, say 3.7.1 ~ 3.7.6 are all merged into one doc, then under folder `3-7-<name>` there exist no sub-dir for docs and only has a single large doc.md, README.md and /img.

# Step 2: Write README summary for each subsections

Write README.md in each section and subsection as summary of that section. See "Principles of writing section summary" for more details.

## Principles of writing section summary

The goal is that contents can be progressively exposed to a user - a user will read the README of a parent node to decide which child nodes to read, and if he selects some child nodes to read further, he will read the README of all **selected** child nodes and further determine whether that child node is truly worth a read. If a leaf node is achieved, the user will read the README of that doc and if the user determines that this doc worths a read, he will read the doc content of that leaf node.

Therefore, write READMEs according to these principles:

1. In the non-leaf-level README, include a 50-word summary for all docs in the whole sub-tree first. Then, include a 50-word summary for each child node. As mentioned above, these concise summaries will be exposed to users so they can decide which child node doc to read. Therefore, provide key points of each child node doc concisely.

2. In the leaf-level README, write a 200-word summary. You can include more details so the user can finally decide whether to read this leaf doc.

# Step 3: Check and fix format

Since all docs are converted from HTML to markdown, go through **all** documentation contents and fix format issues like:

- Character Encoding Issues
- Code Formatting Issues
- Other Issues You Find (Never change the content! **Just change the format**!)
