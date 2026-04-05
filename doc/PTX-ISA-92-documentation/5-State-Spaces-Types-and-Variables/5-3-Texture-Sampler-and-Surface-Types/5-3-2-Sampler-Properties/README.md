# 5.3.2. Sampler Properties

Describes sampler and texture opaque type fields in independent texture mode: `normalized_coords`, `filter_mode` (nearest/linear), `addr_mode_{0,1,2}` (wrap/mirror/clamp variants), array size, mipmap levels, and `force_unnormalized_coords` (samplerref-only, used for OpenCL). Variables of `.texref`, `.samplerref`, `.surfref` types may be declared at module scope (`.global`) or as kernel parameters (`.param`).
