# Mariner model compiler

This package aims to provide model building and make suggestions for the architecture being build.

It works using the [Command](https://refactoring.guru/design-patterns/command) and [Visitor](https://refactoring.guru/design-patterns/visitor).

Commands are implemented in `src/implementation/commands/`, and the model validation is implemented in `src/implementations/validation/`. In that folder, `visitors/` holds the visitors, which are used for model validation. Each visitor implements it's own kind of validation so the code of each one is kept separately.

Types used comes from rtk code generation, and are accurate types for the model schemas accepted by the API.
In fact, the types are so accurate, some times is hard to work with them. This package must be the one with more comments to ignore type errors in the webapp.
