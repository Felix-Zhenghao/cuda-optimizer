#!/usr/bin/env python3
"""
Merge short leaf docs (<300 words) into larger combined docs.
Never merge into a doc that would exceed 2000 words.

Usage:
    python merge.py <doc_dir>

Example:
    python merge.py doc/1-Introduction-PTX-ISA-92-documentation
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def parse_number_prefix(name):
    """Extract the numeric prefix from a directory name.

    Examples:
        "10-1-Special-Registerstid" -> "10-1"
        "11-4-Performance-Tuning-Directives" -> "11-4"
        "8-Memory-Consistency-Model" -> "8"
    Returns (prefix_str, prefix_tuple) or (None, None) if no match.
    """
    m = re.match(r"^((\d+)(-\d+)*)", name)
    if not m:
        return None, None
    prefix_str = m.group(1)
    prefix_tuple = tuple(int(x) for x in prefix_str.split("-"))
    return prefix_str, prefix_tuple


def name_suffix(dirname):
    """Extract the name part after the numeric prefix.

    "10-1-Special-Registerstid" -> "Special-Registerstid"
    "11-4-Performance-Tuning-Directives" -> "Performance-Tuning-Directives"
    """
    m = re.match(r"^(\d+(-\d+)*)-(.*)", dirname)
    if m:
        return m.group(3)
    return dirname


def common_prefix_str(strings):
    """Find the longest common prefix of a list of strings."""
    if not strings:
        return ""
    s0 = strings[0]
    for i, ch in enumerate(s0):
        for s in strings[1:]:
            if i >= len(s) or s[i] != ch:
                return s0[:i]
    return s0


def derive_group_name(dirs):
    """Derive a concise name for a merged group from their directory names."""
    suffixes = [name_suffix(d.name) for d in dirs]
    prefix = common_prefix_str(suffixes).rstrip("-")
    if prefix:
        return prefix
    # Fallback: use first dir's suffix
    return suffixes[0]


def word_count(filepath):
    """Count words in a file."""
    return len(filepath.read_text().split())


def is_leaf(d):
    """A leaf directory has doc.md and no child subdirectories (img/ excluded)."""
    if not (d / "doc.md").exists():
        return False
    child_dirs = [c for c in d.iterdir() if c.is_dir() and c.name != "img"]
    return len(child_dirs) == 0


def get_sorted_leaf_children(parent):
    """Get leaf children of a parent, sorted by numeric prefix."""
    leaves = [c for c in parent.iterdir() if c.is_dir() and is_leaf(c)]
    leaves.sort(key=lambda d: (parse_number_prefix(d.name)[1] or (999,), d.name))
    return leaves


def group_consecutive_short(leaves, threshold=300, max_merged=2000):
    """Group consecutive short leaves, splitting if merged doc would exceed max_merged.

    Returns a list of groups (each a list of Path objects) with 2+ members.
    """
    groups = []
    current_group = []
    current_words = 0

    for leaf in leaves:
        wc = word_count(leaf / "doc.md")
        if wc >= threshold:
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = []
            current_words = 0
        elif current_words + wc > max_merged and current_group:
            # Adding this leaf would exceed the cap — flush current group
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = [leaf]
            current_words = wc
        else:
            current_group.append(leaf)
            current_words += wc

    if len(current_group) >= 2:
        groups.append(current_group)

    return groups


def collect_images(src_dir, dest_img_dir):
    """Collect all images from src_dir's img/ subdir into dest_img_dir."""
    src_img = src_dir / "img"
    if not src_img.exists():
        return
    for img_file in src_img.iterdir():
        if img_file.is_file():
            dest_img_dir.mkdir(exist_ok=True)
            dest = dest_img_dir / img_file.name
            if not dest.exists():
                shutil.move(str(img_file), str(dest))


def merge_group(group, parent_dir):
    """Merge a group of leaf directories into a single directory.

    Creates a new directory with combined doc.md, merged img/ dirs, and README.
    Removes the original directories.
    """
    first_prefix, _ = parse_number_prefix(group[0].name)
    last_prefix, _ = parse_number_prefix(group[-1].name)
    group_name = derive_group_name(group)

    if first_prefix == last_prefix:
        new_name = f"{first_prefix}-{group_name}"
    else:
        new_name = f"{first_prefix}-to-{last_prefix}-{group_name}"

    new_dir = parent_dir / new_name
    new_dir.mkdir(parents=True, exist_ok=True)
    img_dir = new_dir / "img"

    # Concatenate doc.md files and collect images
    combined_md = []
    for leaf in group:
        content = (leaf / "doc.md").read_text().strip()
        combined_md.append(content)
        collect_images(leaf, img_dir)

    (new_dir / "doc.md").write_text("\n\n---\n\n".join(combined_md) + "\n")

    # Write leaf-level README placeholder
    titles = [leaf.name for leaf in group]
    (new_dir / "README.md").write_text(
        f"# {group_name}\n\n"
        f"Merged from: {titles[0]} to {titles[-1]}\n\n"
        "(placeholder - 100-word summary goes here)\n"
    )

    # Remove original directories
    for leaf in group:
        shutil.rmtree(leaf)

    print(f"  Merged {len(group)} docs -> {new_name}")
    return new_dir


def merge_all_into_parent(leaves, parent_dir):
    """When ALL children are short and merged, make parent a leaf node.

    Put the combined doc.md directly in the parent directory.
    """
    img_dir = parent_dir / "img"

    combined_md = []
    for leaf in leaves:
        content = (leaf / "doc.md").read_text().strip()
        combined_md.append(content)
        collect_images(leaf, img_dir)

    (parent_dir / "doc.md").write_text("\n\n---\n\n".join(combined_md) + "\n")

    # Remove original leaf directories
    for leaf in leaves:
        shutil.rmtree(leaf)

    # Rewrite README as leaf-level placeholder
    (parent_dir / "README.md").write_text(
        f"# {name_suffix(parent_dir.name)}\n\n"
        "(placeholder - 100-word summary goes here)\n"
    )

    print(f"  All {len(leaves)} docs merged into parent -> {parent_dir.name} is now a leaf")


def update_parent_readme(parent_dir):
    """Rewrite the parent's README to list its current children."""
    children = sorted(
        [c for c in parent_dir.iterdir() if c.is_dir() and c.name != "img"],
        key=lambda d: (parse_number_prefix(d.name)[1] or (999,), d.name),
    )
    if not children:
        return
    parent_name = name_suffix(parent_dir.name)
    readme = f"# {parent_name}\n\n(placeholder - 30-word summary goes here)\n\n## Contents\n\n"
    for child in children:
        child_name = name_suffix(child.name)
        readme += f"- **{child_name}** — (placeholder - 30-word summary goes here)\n"
    (parent_dir / "README.md").write_text(readme)


def process_parent(parent_dir):
    """Process a single parent directory, merging its short leaf children."""
    leaves = get_sorted_leaf_children(parent_dir)
    if len(leaves) < 2:
        return

    # Check: are ALL leaves short and total under 2000 words?
    word_counts = [(l, word_count(l / "doc.md")) for l in leaves]
    all_short = all(wc < 300 for _, wc in word_counts)
    total_words = sum(wc for _, wc in word_counts)

    # Also check: does the parent have non-leaf children?
    non_leaf_children = [
        c for c in parent_dir.iterdir()
        if c.is_dir() and c.name != "img" and not is_leaf(c)
    ]

    if all_short and not non_leaf_children and total_words <= 2000:
        # All children are short leaves and fit in one doc -> merge into parent
        merge_all_into_parent(leaves, parent_dir)
    else:
        # Merge consecutive groups of short leaves
        groups = group_consecutive_short(leaves)
        for group in groups:
            merge_group(group, parent_dir)
        # Update parent README to reflect new children
        update_parent_readme(parent_dir)


def merge_short_docs(doc_dir):
    """Main entry: merge short docs in the given doc directory tree.

    Processes bottom-up so that deeper merges happen first, potentially
    creating new short parents that can be merged at a higher level.
    """
    doc_root = Path(doc_dir)
    if not doc_root.is_dir():
        print(f"Error: {doc_root} is not a directory")
        sys.exit(1)

    # Collect all non-leaf parent directories, sorted deepest first
    all_dirs = sorted(
        [d for d in doc_root.rglob("*") if d.is_dir()],
        key=lambda d: len(d.parts),
        reverse=True,
    )

    for d in all_dirs:
        if not d.exists():
            continue
        if d == doc_root:
            continue
        if d.name == "img":
            continue

        leaves = get_sorted_leaf_children(d)
        if len(leaves) >= 2:
            print(f"Processing: {d.relative_to(doc_root)}")
            process_parent(d)


def main():
    parser = argparse.ArgumentParser(
        description="Merge short leaf docs (<300 words) into combined docs."
    )
    parser.add_argument("doc_dir", help="Root doc directory to process")
    args = parser.parse_args()

    merge_short_docs(args.doc_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
