# Debugging-Directives

- **11-5-1-Debugging-Directivesdwarf** — `@@DWARF` embeds raw DWARF debug data (bytes, 4-byte, quad) directly in PTX.
- **11-5-2-Debugging-Directivessection** — `.section` creates named DWARF debug sections with structured byte/integer data.
- **11-5-3-Debugging-Directivesfile** — `.file` maps source filenames to numeric indices used by `.loc` directives.
- **11-5-4-Debugging-Directivesloc** — `.loc` associates source file/line/column (with inlining support) with PTX instructions.
