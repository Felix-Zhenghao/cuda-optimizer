---
name: doc-crawl
description: Crawl documents from a given link and integrate into this repo's /doc dir. Use when asked to crawl docs from a link.
---

# Step 1: Explore the document structure of the given link.
For instance, for the cuda programming [doc](https://docs.nvidia.com/cuda/cuda-programming-guide), it has a title "CUDA Programming Guide" and the structure of the first section is:

```
1. Introduction to CUDA
  1.1. Introduction
    1.1.1. The Graphics Processing Unit
    1.1.2. The Benefits of Using GPUs
    1.1.3. Getting Started Quickly
```

# Step 2: Plan with how to integrate the doc in this repo's /doc.

For the above example, in the local /doc, the dir structure should be:

```
/doc/CUDA-Programming-Guide/
├── 1-Introduction-to-CUDA/
│   ├── 1-1-Introduction/
│   │   ├── 1-1-1-The-Graphics-Processing-Unit.md
│   │   ├── 1-1-2-The-Benefits-of-Using-GPUs.md
│   │   ├── 1-1-3-Getting-Started-Quickly.md
│   │   ├── README.md
│   │   └── img/ (if has image)
│   │       └── *.png
│   └── README.md
```

- The dir title should be the same as the doc title of that section with section id like "1-1" added.

- Add README.md in each section and subsection as summary of that section. See "Principles of writing section summary" for more details.

- Store images in dedicated dir in each subsection.

# Principles of writing section summary

The goal is that contents can be progressively exposed to a user - a user will read the README of a section to decide which subsections to read, and if he selects some subsections to read further, he will read the README of all **selected** subsections and further determine whether that subsection is truly worth a read. If so, the user will read the subsection content.

Therefore, write READMEs according to these principles:

1. In the section README, include a 50-word summary of the whole section first. Then, include a 50-word summary of a subsections. As mentioned above, these concise summaries will be exposed to users so they can decide which subsections to read. Therefore, provide key points of each subsection concisely.

2. In the subsection README, write a 200-word summary. You can include more details so the user can finally decide whether to read this subsection

# Some notification

1. Image and its caption (if exists) should always be included. The image should be stored in the dedicated `/img` dir of the subsection that it belongs to. For instance, if "1.1.2. The Benefits of Using GPUs" contains an image with caption "Figure 1 The GPU Devotes More Transistors to Data Processing", then the image should be stored in `/doc/CUDA-Programming-Guide/1-Introduction-to-CUDA/1-1-Introduction/img/Figure-1-The-GPU-Devotes-More-Transistors-to-Data-Processing.png`.
