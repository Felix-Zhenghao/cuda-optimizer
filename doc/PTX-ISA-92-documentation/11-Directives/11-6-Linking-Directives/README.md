# Linking-Directives

Documents linking directives: `.extern` (declares symbol defined in another module), `.visible` (makes symbol globally accessible to other modules; PTX symbols are module-local by default), and `.weak` (visible with weak linkage, allowing override). These enable separate compilation and linking of PTX modules.
