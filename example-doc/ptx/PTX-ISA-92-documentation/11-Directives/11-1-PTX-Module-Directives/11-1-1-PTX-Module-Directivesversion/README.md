# 11.1.1: PTX Module Directives: .version

Specifies the PTX ISA version number (major.minor) for a module. The major number indicates incompatible language changes; the minor number indicates new features. Every PTX module must begin with a .version directive, and only one is allowed per module. Introduced in PTX ISA version 1.0.
