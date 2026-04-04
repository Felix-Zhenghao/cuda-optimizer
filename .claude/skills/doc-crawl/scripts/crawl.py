#!/usr/bin/env python3
"""
Documentation crawler for Sphinx-based documentation sites.
Crawls documentation and organizes into a hierarchical directory structure
with doc.md files for content and placeholder README.md files for summaries.

Usage:
    python crawl.py <url> [-o output_dir]

Example:
    python crawl.py https://docs.nvidia.com/cuda/cuda-programming-guide -o doc
"""

import argparse
from copy import deepcopy
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup, Tag
import markdownify


class DocCrawler:
    def __init__(self, base_url, output_dir):
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; DocCrawler/1.0)"}
        )
        self._page_cache = {}  # URL (without fragment) -> BeautifulSoup

    def fetch(self, url):
        """Fetch a URL and return BeautifulSoup."""
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")

    def download_image(self, url, save_path):
        """Download an image to the specified path."""
        resp = self.session.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

    def slugify(self, text):
        """Convert text to a directory/file-safe slug with hyphens."""
        text = text.strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"\s+", "-", text)
        text = text.strip("-")
        return text

    def section_slug(self, title):
        """Convert section title to slug with section number prefix.

        Examples:
            "1. Introduction to CUDA" -> "1-Introduction-to-CUDA"
            "1.1. Introduction"       -> "1-1-Introduction"
            "1.2. Programming Model"  -> "1-2-Programming-Model"
        """
        match = re.match(r"([\d]+(?:\.[\d]+)*)\.\s*(.*)", title)
        if match:
            num = match.group(1).replace(".", "-")
            name = self.slugify(match.group(2))
            return f"{num}-{name}"
        return self.slugify(title)

    def image_filename(self, alt_text, caption_text, original_url):
        """Generate an image filename from caption/alt text.

        Prefers caption (includes figure number, e.g. "Figure 1 ..."),
        then alt text, then original filename. Truncates to stay under
        filesystem path length limits.
        """
        text = caption_text or alt_text
        if text:
            slug = self.slugify(text)
            if len(slug) > 120:
                slug = slug[:120].rstrip("-")
        else:
            slug = Path(urlparse(original_url).path).stem
        ext = Path(urlparse(original_url).path).suffix or ".png"
        return f"{slug}{ext}"

    def find_first_subpage(self, soup):
        """Find the URL of the first subpage linked from the main page.

        Sphinx main pages often have an empty sidebar TOC. The full TOC
        is available on subpages. This finds a subpage to scrape the
        full sidebar from.
        """
        # Look for next-page link or first content link
        main = self.extract_content(soup)
        if main:
            for a in main.find_all("a", href=True):
                href = a["href"]
                if (
                    href.endswith(".html")
                    and not href.startswith("http")
                    and not href.startswith("#")
                    and "contents" not in href
                ):
                    return urljoin(self.base_url + "/", href)

        # Fallback: look in nav for any subpage link
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".html") and "/" in href and not href.startswith("http"):
                return urljoin(self.base_url + "/", href)
        return None

    def parse_toc(self, soup):
        """Parse the sidebar TOC using toctree CSS classes.

        Sphinx uses classes like toctree-l1, toctree-l2, toctree-l3 to
        indicate nesting level. This is more reliable than relying on
        nested <ul> structure.

        If the sidebar on this page is empty/shallow, tries fetching a
        subpage to get the full sidebar.
        """
        doc_title, sections = self._extract_toc_from_sidebar(soup)

        # If we only got the title with no real sections, try a subpage
        if not sections:
            subpage_url = self.find_first_subpage(soup)
            if subpage_url:
                print(f"Fetching subpage for full TOC: {subpage_url}")
                sub_soup = self.fetch(subpage_url)
                doc_title, sections = self._extract_toc_from_sidebar(sub_soup)

        if not sections:
            return None

        return {"title": doc_title, "sections": sections}

    def _extract_toc_from_sidebar(self, soup):
        """Extract TOC structure from sidebar navigation.

        Returns (doc_title, sections_list).
        Recursively parses all toctree-lN levels to build an arbitrary-depth tree.
        """
        nav = soup.find("nav", class_="bd-docs-nav")
        if not nav:
            nav = soup.find("nav", class_="wy-nav-side")
        if not nav:
            nav = soup

        # Find all <li> with toctree-lN classes
        all_lis = nav.find_all("li", class_=re.compile(r"toctree-l\d+"))
        if not all_lis:
            return ("Documentation", [])

        # Determine the level of each <li> item
        def get_level(li):
            for cls in li.get("class", []):
                m = re.match(r"toctree-l(\d+)", cls)
                if m:
                    return int(m.group(1))
            return 0

        # Check if l1 items are the doc title or actual sections
        l1_items = [li for li in all_lis if "toctree-l1" in li.get("class", [])]

        if len(l1_items) == 1:
            # Single l1 item = doc title; sections start at l2
            doc_title = "Documentation"
            a = l1_items[0].find("a", recursive=False)
            if a:
                doc_title = a.get_text(strip=True)
            min_level = 2  # skip l1
            base_parent_level = 1
        else:
            # Multiple l1 items = they ARE the sections; get doc title from page
            doc_title = "Documentation"
            title_tag = soup.find("title")
            if title_tag:
                raw = title_tag.get_text(strip=True)
                raw = re.split(r"\s*[—–\-\|]\s*", raw)[0].strip()
                if raw:
                    doc_title = raw
            min_level = 1  # include l1 items as sections
            base_parent_level = 0

        # Parse items into list of (level, title, url)
        items = []
        for li in all_lis:
            level = get_level(li)
            if level < min_level:
                continue
            a = li.find("a", recursive=False)
            if not a:
                continue
            title = a.get_text(strip=True)
            href = a.get("href", "")
            url = urljoin(self.base_url + "/", href) if href and href != "#" else ""
            items.append((level, title, url))

        # Build tree recursively from flat list
        def build_tree(items, start, parent_level):
            """Build children list from items[start:] that are deeper than parent_level.
            Returns (children_list, next_index)."""
            children = []
            i = start
            while i < len(items):
                level, title, url = items[i]
                if level <= parent_level:
                    break  # Back to parent or sibling of parent
                if level == parent_level + 1:
                    node = {"title": title, "url": url, "children": []}
                    # Look ahead for children of this node
                    sub_children, i = build_tree(items, i + 1, level)
                    node["children"] = sub_children
                    children.append(node)
                else:
                    # Skip unexpected deeper items without a parent at the right level
                    i += 1
            return children, i

        sections, _ = build_tree(items, 0, base_parent_level)
        return (doc_title, sections)

    def extract_content(self, soup):
        """Extract the main content div from a page."""
        for selector in [
            ("article", {}),
            ("div", {"role": "main"}),
            ("main", {}),
            ("div", {"class": "body"}),
            ("div", {"class": "document"}),
        ]:
            content = soup.find(selector[0], selector[1])
            if content:
                return content
        return None

    def process_images(self, content, page_url, img_dir):
        """Download images, update src to local paths, clean up wrappers."""
        # Collect figures first to avoid modifying tree while iterating
        figures = content.find_all("figure")
        for figure in figures:
            img = figure.find("img")
            if not img:
                continue
            src = img.get("src", "")
            if not src:
                continue

            img_url = urljoin(page_url, src)
            alt = img.get("alt", "")
            caption = ""
            figcaption = figure.find("figcaption")
            if figcaption:
                # Remove permalink anchors before extracting text
                for plink in figcaption.find_all("a", class_="headerlink"):
                    plink.decompose()
                caption = figcaption.get_text(strip=True)
                # Fix missing space after figure number: "Figure 1Title" -> "Figure 1 Title"
                caption = re.sub(r"(Figure\s+\d+)([A-Z])", r"\1 \2", caption)

            filename = self.image_filename(alt, caption, img_url)
            local_path = img_dir / filename

            try:
                self.download_image(img_url, local_path)
            except Exception as e:
                print(f"    Warning: Failed to download image {img_url}: {e}")
                continue

            # Replace entire <figure> with clean HTML for markdownify
            caption_html = f"\n<p><em>{caption}</em></p>" if caption else ""
            new_html = (
                f'<img src="img/{filename}" alt="{alt}" />'
                f"{caption_html}"
            )
            figure.replace_with(BeautifulSoup(new_html, "html.parser"))

        # Handle standalone images (not inside figures)
        for img in content.find_all("img"):
            src = img.get("src", "")
            if not src or src.startswith("img/"):
                continue  # Already processed
            width = img.get("width", "")
            if width and width.isdigit() and int(width) < 20:
                continue

            img_url = urljoin(page_url, src)
            alt = img.get("alt", "")
            filename = self.image_filename(alt, "", img_url)
            local_path = img_dir / filename

            try:
                self.download_image(img_url, local_path)
                img["src"] = f"img/{filename}"
                # Unwrap <a> wrapper if present
                parent_a = img.find_parent("a")
                if parent_a and parent_a.find("img") == img:
                    parent_a.replace_with(img)
            except Exception as e:
                print(f"    Warning: Failed to download image {img_url}: {e}")

    def content_to_markdown(self, content):
        """Convert a BeautifulSoup content element to clean markdown."""
        # Remove navigation elements
        for tag in content.find_all(["nav", "footer"]):
            tag.decompose()
        for tag in content.find_all(
            "div",
            class_=re.compile(
                r"(prev|next|breadcrumb|header-links|pagination|topbar)"
            ),
        ):
            tag.decompose()
        # Remove permalink anchors (the paragraph symbols)
        for tag in content.find_all("a", class_="headerlink"):
            tag.decompose()

        md = markdownify.markdownify(
            str(content),
            heading_style="ATX",
            strip=["script", "style"],
        )

        # Clean up
        md = re.sub(r"\n{3,}", "\n\n", md)
        lines = [line.rstrip() for line in md.split("\n")]
        md = "\n".join(lines)
        md = md.strip()
        return md

    def crawl(self):
        """Main entry point: crawl the documentation site."""
        print(f"Fetching main page: {self.base_url}")
        soup = self.fetch(self.base_url)

        toc = self.parse_toc(soup)
        if not toc:
            print("Error: Could not parse table of contents from the page.")
            sys.exit(1)

        doc_title = toc["title"]
        doc_slug = self.slugify(doc_title)
        doc_dir = self.output_dir / doc_slug

        print(f"Document: {doc_title}")
        print(f"Output: {doc_dir}")
        print(f"Found {len(toc['sections'])} top-level sections")

        for section in toc["sections"]:
            self._process_node(section, doc_dir)

        # Collapse single-child directories
        self._collapse_single_children(doc_dir)

        print("\nDone! Documentation saved to:", doc_dir)

    def _process_node(self, node, parent_dir, depth=0):
        """Recursively process a TOC node at any depth.

        Nodes with children become directories with a README listing children.
        Leaf nodes get their page crawled and saved as doc.md.
        """
        title = node["title"]
        slug = self.section_slug(title)
        node_dir = parent_dir / slug

        indent = "  " * depth
        print(f"{indent}{'Section' if node['children'] else 'Leaf'}: {title}")

        node_dir.mkdir(parents=True, exist_ok=True)

        if node["children"]:
            # Non-leaf: write README with child listing, recurse into children
            readme = f"# {title}\n\n"
            for child in node["children"]:
                readme += (
                    f"- **{child['title']}** — "
                    "(placeholder - 20-word summary goes here)\n"
                )
            (node_dir / "README.md").write_text(readme)

            for child in node["children"]:
                self._process_node(child, node_dir, depth + 1)
        else:
            # Leaf: crawl content
            self._crawl_page(node["url"], node_dir)
            (node_dir / "README.md").write_text(
                f"# {title}\n\n"
                "(placeholder - 50-word summary goes here)\n"
            )

    def _collapse_single_children(self, root_dir):
        """Collapse directories that have exactly one child subdirectory.

        When a parent has only one child dir (and no doc.md of its own),
        the child's contents are moved into the parent, eliminating the
        unnecessary nesting level. Processes bottom-up so nested single-child
        chains are fully collapsed.
        """
        all_dirs = sorted(
            [d for d in root_dir.rglob("*") if d.is_dir()],
            key=lambda d: len(d.parts),
            reverse=True,
        )

        for dir_path in all_dirs:
            if not dir_path.exists():
                continue

            child_dirs = [d for d in dir_path.iterdir() if d.is_dir() and d.name != "img"]
            has_doc = (dir_path / "doc.md").exists()

            if len(child_dirs) == 1 and not has_doc:
                child = child_dirs[0]
                print(f"Collapsing: {dir_path.name}/{child.name} -> {dir_path.name}")

                for item in child.iterdir():
                    dest = dir_path / item.name
                    if dest.exists():
                        if item.name == "README.md":
                            dest.unlink()
                        else:
                            continue
                    shutil.move(str(item), str(dest))

                child.rmdir()

    def _extract_section_content(self, content, fragment):
        """Extract only the content for a specific fragment/anchor from a page.

        Finds the element with the given id, then collects all sibling content
        until the next heading of the same or higher level.
        """
        # Find the target element by id
        target = content.find(id=fragment)
        if not target:
            # Try finding an anchor with that name
            target = content.find("a", attrs={"name": fragment})
        if not target:
            return None

        # Find the heading element - it might be the target itself or its parent
        heading = None
        if target.name and re.match(r"h[1-6]", target.name):
            heading = target
        else:
            # The id might be on a <section> or <div> wrapper - if so, return it directly
            if target.name in ("section", "div"):
                return deepcopy(target)
            # Or the id is on a span/anchor inside a heading
            parent = target.parent
            if parent and parent.name and re.match(r"h[1-6]", parent.name):
                heading = parent

        if not heading:
            # If we found the target but can't identify a heading structure,
            # check if it's a section wrapper
            return None

        heading_level = int(heading.name[1])

        # Collect the heading and all following siblings until next same/higher level heading
        collected = []
        collected.append(deepcopy(heading))

        for sibling in heading.next_siblings:
            if not isinstance(sibling, Tag):
                # Keep NavigableString (whitespace, text nodes)
                collected.append(deepcopy(sibling))
                continue
            # Stop at next heading of same or higher level
            if sibling.name and re.match(r"h[1-6]", sibling.name):
                sib_level = int(sibling.name[1])
                if sib_level <= heading_level:
                    break
            collected.append(deepcopy(sibling))

        # Build a new soup with just the collected elements
        wrapper = BeautifulSoup("<div></div>", "html.parser").div
        for elem in collected:
            wrapper.append(elem)
        return wrapper

    def _crawl_page(self, url, target_dir):
        """Fetch a page, extract content, download images, save as doc.md."""
        if not url:
            (target_dir / "doc.md").write_text("# Error\n\nNo URL provided.\n")
            return

        img_dir = target_dir / "img"

        # Split URL and fragment
        base_url, fragment = urldefrag(url)

        # Use cached page if available
        if base_url in self._page_cache:
            soup = self._page_cache[base_url]
        else:
            try:
                soup = self.fetch(base_url)
                self._page_cache[base_url] = soup
            except Exception as e:
                print(f"    Warning: Failed to fetch {base_url}: {e}")
                (target_dir / "doc.md").write_text(
                    f"# Error\n\nFailed to fetch content from {url}\n"
                )
                return

        content = self.extract_content(soup)
        if not content:
            print(f"    Warning: No content found at {url}")
            (target_dir / "doc.md").write_text(
                f"# Error\n\nNo content found at {url}\n"
            )
            return

        # If there's a fragment, extract only that section's content
        if fragment:
            section_content = self._extract_section_content(content, fragment)
            if section_content:
                content = section_content
            else:
                print(f"    Warning: Fragment #{fragment} not found, using full page")

        # Work on a deep copy to avoid mutating the cached page
        content = deepcopy(content)

        # Download images and update references
        self.process_images(content, base_url, img_dir)

        # Convert to markdown
        md = self.content_to_markdown(content)

        (target_dir / "doc.md").write_text(md + "\n")

        # Clean up empty img dir
        if img_dir.exists() and not any(img_dir.iterdir()):
            img_dir.rmdir()


def main():
    parser = argparse.ArgumentParser(
        description="Crawl documentation from a URL into structured markdown."
    )
    parser.add_argument("url", help="URL of the documentation to crawl")
    parser.add_argument(
        "-o",
        "--output",
        default="doc",
        help="Output directory (default: doc)",
    )
    args = parser.parse_args()

    crawler = DocCrawler(args.url, args.output)
    crawler.crawl()


if __name__ == "__main__":
    main()
