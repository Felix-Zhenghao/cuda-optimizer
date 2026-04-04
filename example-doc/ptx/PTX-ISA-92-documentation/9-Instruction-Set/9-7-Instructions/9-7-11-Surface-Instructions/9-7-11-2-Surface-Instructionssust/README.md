# 9.7.11.2. Surface Instructions: sust

The sust instruction stores data to surface memory. Supports unformatted (sust.b) and formatted (sust.p) modes across 1d/2d/3d surfaces and surface arrays. Formatted mode interprets source as RGBA components. Provides out-of-bounds handling via trap, clamp, or zero modifiers.
