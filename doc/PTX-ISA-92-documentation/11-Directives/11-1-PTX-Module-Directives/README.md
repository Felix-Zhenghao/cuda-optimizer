# PTX-Module-Directives

- **11-1-1-PTX-Module-Directivesversion** — `.version` directive specifying PTX ISA major.minor version; each module must begin with exactly one.
- **11-1-2-PTX-Module-Directivestarget** — `.target` directive setting GPU architecture (with a/f suffix variants) and platform options like texture mode.
- **11-1-3-PTX-Module-Directivesaddress_size** — `.address_size` directive (32 or 64 bits) for module-wide pointer width; defaults to 32.
