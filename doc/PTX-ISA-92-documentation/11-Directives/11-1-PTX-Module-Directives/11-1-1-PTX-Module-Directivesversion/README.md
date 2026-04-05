# 11.1.1. PTX Module Directives:.version

Defines `.version` directive specifying PTX ISA version number (major.minor). Major increments on breaking changes; minor on new features. Each PTX module must begin with exactly one `.version` directive. Compilation tools must support equal or greater version than specified.
