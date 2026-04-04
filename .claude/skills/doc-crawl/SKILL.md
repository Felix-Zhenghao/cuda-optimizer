---
name: doc-crawl
description: Crawl documents from a given link and integrate into this repo's /doc dir. Use when asked to crawl docs from a link.
model: sonnet
effort: low
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# Step 1: Run `scripts/run.sh` with the given link to crawl the doc.

Usage: `bash run.sh <url> [-o output_dir]`

# Step 2: Write README summary for each subsections

Write README.md in each section and subsection as summary of that section. See "Principles of writing section summary" for more details.

## Principles of writing section summary

The goal is that contents can be progressively exposed to a user - a user will read the README of a parent node to decide which child nodes to read, and if he selects some child nodes to read further, he will read the README of all **selected** child nodes and further determine whether that child node is truly worth a read. If a leaf node is achieved, the user will read the README of that doc and if the user determines that this doc worths a read, he will read the doc content of that leaf node.

Therefore, write READMEs according to these principles:

1. In the non-leaf-level README include a 20-word summary for each child node. As mentioned above, these concise summaries will be exposed to users so they can decide which child node doc to read. Therefore, provide key points of each child node doc concisely.

2. In the leaf-level README, write a 50-word summary. You can include more details so the user can finally decide whether to read this leaf doc.

# Step 3: Check and fix format

Since all docs are converted from HTML to markdown, go through **all** documentation contents and fix format issues like:

- Character Encoding Issues
- Code Formatting Issues
- Other Issues You Find (Never change the content! **Just change the format**!)

# FINAL NOTIFICATION

1. Write all summaries by first reading the doc content and then summarize by yourself. NEVER write script to do this.
2. When fix format issues, you should read the whole doc content and fix them. Don't write python script to do this. You can fix during reading when summarizing things so you only need to read doc content once.
3. DON'T BE LAZY. After you **have** read a doc, written the summary and fixed all format issues for this doc, add the doc dir name to the a tmp file. WHEN YOU THINK YOU HAVE COMPLETED ALL TASKS, ALAWYS CHECK THIS TMP FILE TO MAKE SURE ALL DOC FILES ARE ON IT.
4. If target dir is not specified, prompt the user to fill in the dir to store the doc. For instance, "/doc".
5. Launch parallel agents to accelerate summary writing and format issue fixing.
