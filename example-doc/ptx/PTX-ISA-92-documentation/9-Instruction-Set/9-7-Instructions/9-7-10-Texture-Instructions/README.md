# Texture Instructions

- **Texturing-Modes** -- Unified vs independent texturing modes controlling sampler/texture handle associations and limits.
- **Mipmaps** -- Mipmap concepts and PTX support for base, level, and gradient LOD selection modes.
- **tex** -- Texture memory lookup instruction supporting multiple geometries, mipmaps, and texturing modes.
- **tld4** -- Fetches 4-texel bilinear footprint for texture filtering with multiple geometry support.
- **txq** -- Queries texture and sampler attributes like dimensions, format, and filtering properties.
- **istypep** -- Predicate query checking whether a register points to a texref, samplerref, or surfref.
